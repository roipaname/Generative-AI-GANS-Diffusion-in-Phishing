import json
import os
from datetime import datetime
from pathlib import Path

PROCESSED_DATA_DIR = Path("./processed_data")
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

def save_analysis_data(analysis_type, text=None, image_result=None, text_result=None, 
                       combined_result=None, urls_found=None, url_results=None, 
                       voice_data=None):
    """
    Save analysis data to processed_data directory as timestamped JSON
    
    Args:
        analysis_type: str - 'text', 'image', 'combined', 'voice'
        text: str - analyzed text content
        image_result: dict - image analysis results
        text_result: dict - text analysis results
        combined_result: dict - combined analysis results
        urls_found: list - URLs detected in text
        url_results: list - VirusTotal scan results for URLs
        voice_data: dict - voice recording metadata
        
    Returns:
        str: filepath of saved JSON file, or None if error
    """
    
    timestamp = datetime.now()
    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
    filepath = PROCESSED_DATA_DIR / filename
    
    # Determine if phishing based on results
    is_phishing = False
    confidence = 0.0
    overall_verdict = "legitimate"
    
    if combined_result:
        label = combined_result.get('label', '').lower()
        is_phishing = 'phishing' in label and 'not phishing' not in label
        confidence = combined_result.get('confidence', 0.0)
        overall_verdict = label
    elif text_result:
        label = text_result.get('label', '').lower()
        is_phishing = 'phishing' in label and 'not phishing' not in label
        confidence = text_result.get('confidence', 0.0)
        overall_verdict = label
    elif image_result:
        label = image_result.get('label', '').lower()
        is_phishing = 'phishing' in label and 'not phishing' not in label
        confidence = image_result.get('confidence', 0.0)
        overall_verdict = label
    
    # Check URL results for threats
    malicious_urls = 0
    suspicious_urls = 0
    url_threat_details = []
    
    if url_results:
        for idx, url_res in enumerate(url_results):
            if url_res:
                verdict = url_res.get('verdict', '').lower()
                url_info = {
                    'url': urls_found[idx] if urls_found and idx < len(urls_found) else 'unknown',
                    'verdict': verdict,
                    'malicious': url_res.get('malicious', 0),
                    'suspicious': url_res.get('suspicious', 0),
                    'harmless': url_res.get('harmless', 0)
                }
                url_threat_details.append(url_info)
                
                if 'malicious' in verdict or 'phishing' in verdict or 'Malicious' in verdict:
                    malicious_urls += 1
                    is_phishing = True  # Mark as phishing if any URL is malicious
                elif 'suspicious' in verdict or 'Suspicious' in verdict:
                    suspicious_urls += 1
    
    # Calculate risk level
    risk_level = "low"
    if is_phishing:
        if confidence >= 0.8 or malicious_urls > 0:
            risk_level = "high"
        elif confidence >= 0.5 or suspicious_urls > 0:
            risk_level = "medium"
        else:
            risk_level = "low"
    
    # Build comprehensive data structure
    data = {
        # Basic metadata
        'timestamp': timestamp.isoformat(),
        'scan_type': analysis_type,
        'is_phishing': is_phishing,
        'confidence': confidence,
        'risk_level': risk_level,
        'overall_verdict': overall_verdict,
        
        # Content data
        'text_content': text[:1000] if text else None,  # Store first 1000 chars
        'text_length': len(text) if text else 0,
        'has_image': image_result is not None,
        
        # Analysis results
        'text_analysis': {
            'label': text_result.get('label') if text_result else None,
            'confidence': text_result.get('confidence') if text_result else None
        } if text_result else None,
        
        'image_analysis': {
            'label': image_result.get('label') if image_result else None,
            'confidence': image_result.get('confidence') if image_result else None
        } if image_result else None,
        
        'combined_analysis': {
            'label': combined_result.get('label') if combined_result else None,
            'confidence': combined_result.get('confidence') if combined_result else None,
            'text': combined_result.get('text') if combined_result else None,
            'image': combined_result.get('image') if combined_result else None
        } if combined_result else None,
        
        # URL data
        'urls_detected': len(urls_found) if urls_found else 0,
        'urls': urls_found if urls_found else [],
        'malicious_urls': malicious_urls,
        'suspicious_urls': suspicious_urls,
        'url_threat_details': url_threat_details,
        
        # Voice data
        'voice_metadata': voice_data if voice_data else None,
        
        # Additional metadata
        'scan_timestamp_readable': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'scan_date': timestamp.strftime('%Y-%m-%d'),
        'scan_time': timestamp.strftime('%H:%M:%S')
    }
    
    # Write to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Analysis data saved: {filepath}")
        return str(filepath)
    except Exception as e:
        print(f"❌ Error saving analysis data: {e}")
        return None


def load_scan_data(filename):
    """
    Load a specific scan data file
    
    Args:
        filename: str - name of the JSON file to load
        
    Returns:
        dict: scan data or None if error
    """
    filepath = PROCESSED_DATA_DIR / filename
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading scan data from {filename}: {e}")
        return None


def get_all_scans(limit=None):
    """
    Get all scan filenames sorted by date (newest first)
    
    Args:
        limit: int - maximum number of scans to return (None for all)
        
    Returns:
        list: list of scan filenames
    """
    try:
        files = sorted(PROCESSED_DATA_DIR.glob("*.json"), reverse=True)
        filenames = [f.name for f in files]
        if limit:
            return filenames[:limit]
        return filenames
    except Exception as e:
        print(f"❌ Error getting scan list: {e}")
        return []


def get_scan_statistics():
    """
    Calculate comprehensive statistics from all scans
    
    Returns:
        dict: statistics dictionary with various metrics
    """
    scans = get_all_scans()
    
    stats = {
        'total_scans': len(scans),
        'phishing_count': 0,
        'legitimate_count': 0,
        'total_urls_scanned': 0,
        'malicious_urls': 0,
        'suspicious_urls': 0,
        'scan_types': {
            'text': 0, 
            'image': 0, 
            'combined': 0, 
            'voice': 0
        },
        'risk_levels': {
            'high': 0,
            'medium': 0,
            'low': 0
        },
        'average_confidence': 0.0,
        'scans_by_date': {},
        'threats_by_date': {}
    }
    
    total_confidence = 0.0
    confidence_count = 0
    
    for scan_file in scans:
        data = load_scan_data(scan_file)
        if data:
            # Basic counts
            if data.get('is_phishing'):
                stats['phishing_count'] += 1
            else:
                stats['legitimate_count'] += 1
            
            # URL stats
            stats['total_urls_scanned'] += data.get('urls_detected', 0)
            stats['malicious_urls'] += data.get('malicious_urls', 0)
            stats['suspicious_urls'] += data.get('suspicious_urls', 0)
            
            # Scan type stats
            scan_type = data.get('scan_type', 'unknown')
            if scan_type in stats['scan_types']:
                stats['scan_types'][scan_type] += 1
            
            # Risk level stats
            risk_level = data.get('risk_level', 'low')
            if risk_level in stats['risk_levels']:
                stats['risk_levels'][risk_level] += 1
            
            # Confidence average
            confidence = data.get('confidence', 0)
            if confidence > 0:
                total_confidence += confidence
                confidence_count += 1
            
            # Date-based stats
            scan_date = data.get('scan_date', 'unknown')
            if scan_date != 'unknown':
                stats['scans_by_date'][scan_date] = stats['scans_by_date'].get(scan_date, 0) + 1
                if data.get('is_phishing'):
                    stats['threats_by_date'][scan_date] = stats['threats_by_date'].get(scan_date, 0) + 1
    
    # Calculate average confidence
    if confidence_count > 0:
        stats['average_confidence'] = total_confidence / confidence_count
    
    return stats


def get_recent_threats(limit=10):
    """
    Get recent phishing threats detected
    
    Args:
        limit: int - number of recent threats to return
        
    Returns:
        list: list of threat data dictionaries
    """
    scans = get_all_scans()
    threats = []
    
    for scan_file in scans:
        if len(threats) >= limit:
            break
            
        data = load_scan_data(scan_file)
        if data and data.get('is_phishing'):
            threats.append({
                'timestamp': data.get('timestamp'),
                'scan_type': data.get('scan_type'),
                'confidence': data.get('confidence'),
                'risk_level': data.get('risk_level'),
                'urls_detected': data.get('urls_detected', 0),
                'malicious_urls': data.get('malicious_urls', 0),
                'text_preview': data.get('text_content', '')[:100] if data.get('text_content') else None
            })
    
    return threats


def cleanup_old_scans(days_to_keep=30):
    """
    Delete scan data older than specified days
    
    Args:
        days_to_keep: int - number of days of data to keep
        
    Returns:
        int: number of files deleted
    """
    try:
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        files = PROCESSED_DATA_DIR.glob("*.json")
        deleted_count = 0
        
        for file in files:
            if file.stat().st_mtime < cutoff_date:
                file.unlink()
                deleted_count += 1
                print(f"🗑️ Deleted old scan: {file.name}")
        
        print(f"✅ Cleanup complete: {deleted_count} files deleted")
        return deleted_count
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return 0


def export_scans_to_csv(output_file="scans_export.csv"):
    """
    Export all scan data to CSV file
    
    Args:
        output_file: str - output CSV filename
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import csv
        scans = get_all_scans()
        
        if not scans:
            print("⚠️ No scans to export")
            return False
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'scan_type', 'is_phishing', 'confidence', 
                         'risk_level', 'urls_detected', 'malicious_urls']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for scan_file in scans:
                data = load_scan_data(scan_file)
                if data:
                    writer.writerow({
                        'timestamp': data.get('timestamp'),
                        'scan_type': data.get('scan_type'),
                        'is_phishing': data.get('is_phishing'),
                        'confidence': data.get('confidence'),
                        'risk_level': data.get('risk_level'),
                        'urls_detected': data.get('urls_detected', 0),
                        'malicious_urls': data.get('malicious_urls', 0)
                    })
        
        print(f"✅ Scans exported to {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error exporting scans: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    print("🛡️ PhishGuard Data Logger - Test Mode\n")
    
    # Test saving data
    print("📝 Testing data save...")
    test_result = {
        'label': 'phishing',
        'confidence': 0.92
    }
    
    filepath = save_analysis_data(
        analysis_type='text',
        text='This is a test phishing email. Click here: http://suspicious-site.com',
        text_result=test_result,
        urls_found=['http://suspicious-site.com'],
        url_results=[{
            'verdict': 'malicious',
            'malicious': 5,
            'suspicious': 2,
            'harmless': 0
        }]
    )
    
    if filepath:
        print(f"✅ Test data saved to: {filepath}\n")
    
    # Get statistics
    print("📊 Current Statistics:")
    stats = get_scan_statistics()
    print(f"   Total Scans: {stats['total_scans']}")
    print(f"   Phishing: {stats['phishing_count']}")
    print(f"   Legitimate: {stats['legitimate_count']}")
    print(f"   Average Confidence: {stats['average_confidence']:.2%}")
    print(f"   Total URLs Scanned: {stats['total_urls_scanned']}")
    print(f"   Malicious URLs: {stats['malicious_urls']}\n")
    
    print("✅ Data logger module ready!")