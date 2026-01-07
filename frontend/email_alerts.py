"""
Email Alert System for PhishGuard AI
Sends email notifications when threats are detected
UI-matched styling with modern glassmorphic design
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class EmailAlerter:
    def __init__(self):
        """Initialize email alerter with SMTP configuration"""
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = "ebebeclarence920@gmail.com"
        self.sender_password = "ktqj ubtc iscl czem"
        self.alert_recipients = "ebebeclarence55@gmail.com,clarence@kodah.ai".split(",")
    def auto_send_if_critical(self, result: Dict[str, tuple]) -> None:
      """
      Automatically sends an alert email if any modality (text/image/combined)
      is classified as 'phishing' with confidence >= 99.9%.
      """
      for source, (label, confidence) in result.items():
        label = label.lower()
        confidence = float(confidence)

        if label == "phishing" and confidence >= 0.999:
            print(f"🚨 High-confidence phishing detected in {source}! ({confidence*100:.2f}%)")
            threat_data = {
                "label": label,
                "confidence": confidence,
                "source": source,
            }
            self.send_threat_alert(threat_data, severity="CRITICAL")
            return  # Stop after first critical alert
    print("✅ No high-confidence phishing detected; no email sent.")


        
    def send_threat_alert(
        self, 
        threat_data: Dict, 
        severity: str = "HIGH",
        attach_report: Optional[str] = None
    ) -> bool:
        """
        Send email alert for detected threat
        
        Args:
            threat_data: Dictionary containing threat information
            severity: Threat severity level (LOW, MEDIUM, HIGH, CRITICAL)
            attach_report: Path to report file to attach (optional)
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(self.alert_recipients)
            msg['Subject'] = f"🚨 PhishGuard Alert: {severity} Severity Threat Detected"
            
            # Email body
            body = self._create_alert_body(threat_data, severity)
            msg.attach(MIMEText(body, 'html'))
            
            # Attach report if provided
            if attach_report and os.path.exists(attach_report):
                with open(attach_report, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {os.path.basename(attach_report)}",
                    )
                    msg.attach(part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"✅ Alert email sent to {len(self.alert_recipients)} recipient(s)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send alert email: {str(e)}")
            return False
    
    def _create_alert_body(self, threat_data: Dict, severity: str) -> str:
        """Create HTML email body with UI-matched styling"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine alert styling based on severity
        is_phishing = threat_data.get('label', '').lower() == 'phishing'
        
        if is_phishing:
            alert_bg = "rgba(255, 77, 77, 0.15)"
            alert_border = "#ff4d4d"
            alert_glow = "0 0 20px rgba(255, 77, 77, 0.7)"
            status_emoji = "🚨"
            status_text = "THREAT DETECTED"
        else:
            alert_bg = "rgba(100, 255, 218, 0.15)"
            alert_border = "#64ffda"
            alert_glow = "0 0 20px rgba(100, 255, 218, 0.7)"
            status_emoji = "✅"
            status_text = "LEGITIMATE"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                    background: radial-gradient(circle at 20% 80%, rgba(0, 180, 255, 0.15) 0%, transparent 50%),
                                radial-gradient(circle at 80% 20%, rgba(0, 200, 255, 0.15) 0%, transparent 50%),
                                linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #334155 100%);
                    min-height: 100vh;
                    color: #e2e8f0;
                }}
            </style>
        </head>
        <body style="margin: 0; padding: 40px 20px;">
            <div style="max-width: 700px; margin: 0 auto;">
                
                <!-- Header Card -->
                <div style="text-align: center; margin-bottom: 3rem; padding: 3rem 2rem; 
                            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
                            border-radius: 32px; border: 2px solid rgba(59, 130, 246, 0.2);
                            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);">
                    <h1 style="font-family: 'Playfair Display', serif; font-size: 3rem; font-weight: 800; margin: 0;
                               background: linear-gradient(135deg, #64ffda 0%, #38bdf8 40%, #2563eb 100%);
                               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                               background-clip: text;">
                        PhishGuard AI
                    </h1>
                    <p style="font-size: 1.1rem; color: #94a3b8; margin: 1rem 0 0 0; font-weight: 500;">
                        Advanced Threat Detection System
                    </p>
                </div>
                
                <!-- Alert Status Card -->
                <div style="background: {alert_bg}; border: 2px solid {alert_border};
                            box-shadow: {alert_glow}; border-radius: 28px; padding: 2.5rem;
                            margin-bottom: 2rem;">
                    <div style="text-align: center; margin-bottom: 1.5rem;">
                        <span style="font-size: 3rem;">{status_emoji}</span>
                        <h2 style="font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0; color: #f8fafc;">
                            {status_text}
                        </h2>
                        <p style="color: #cbd5e1; margin: 0.5rem 0;">
                            Severity: <strong>{severity}</strong>
                        </p>
                        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0.3rem 0;">
                            {timestamp}
                        </p>
                    </div>
                </div>
                
                <!-- Threat Details Card -->
                <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%);
                            backdrop-filter: blur(25px); border-radius: 28px; padding: 2.5rem;
                            margin-bottom: 2rem; border: 2px solid rgba(59, 130, 246, 0.2);
                            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);">
                    <h3 style="color: #f8fafc; font-size: 1.4rem; margin-top: 0; margin-bottom: 1.5rem; font-weight: 700;">
                        📊 Detection Details
                    </h3>
                    
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 12px 0; color: #94a3b8; font-weight: 600;">Classification:</td>
                            <td style="padding: 12px 0; color: #e2e8f0; font-weight: 600;">
                                {threat_data.get('label', 'Unknown').title()}
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 12px 0; color: #94a3b8; font-weight: 600;">Confidence:</td>
                            <td style="padding: 12px 0; color: #e2e8f0; font-weight: 600;">
                                {threat_data.get('confidence', 0):.1%}
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 12px 0; color: #94a3b8; font-weight: 600;">Content Type:</td>
                            <td style="padding: 12px 0; color: #e2e8f0; font-weight: 600;">
                                {threat_data.get('content_type', 'Mixed')}
                            </td>
                        </tr>
                    </table>
                </div>
                
                {self._format_content_preview(threat_data)}
                
                {self._format_url_results(threat_data.get('urls', []))}
                
                <!-- Action Button -->
                <div style="text-align: center; margin: 2.5rem 0;">
                    <a href="#" style="display: inline-block; 
                                      background: linear-gradient(135deg, #64ffda 0%, #38bdf8 50%, #2563eb 100%);
                                      color: #f8fafc; text-decoration: none; border-radius: 16px;
                                      padding: 1rem 2.5rem; font-weight: 700;
                                      box-shadow: 0 12px 24px rgba(37, 99, 235, 0.4);">
                        View Full Report
                    </a>
                </div>
                
                <!-- Footer -->
                <div style="text-align: center; margin-top: 3rem; padding-top: 2rem;
                            border-top: 2px solid rgba(59, 130, 246, 0.2); color: #94a3b8; font-size: 0.9rem;">
                    <p style="margin: 0.5rem 0;">This is an automated alert from PhishGuard AI</p>
                    <p style="margin: 0.5rem 0;">Advanced AI-Powered Phishing Detection</p>
                    <p style="margin: 1rem 0 0 0; font-size: 0.8rem; color: #64748b;">
                        © 2025 PhishGuard AI. All rights reserved.
                    </p>
                </div>
                
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_content_preview(self, threat_data: Dict) -> str:
        """Format content preview section with UI styling"""
        text_preview = threat_data.get('text_preview', '')
        if not text_preview:
            return ""
        
        if len(text_preview) > 200:
            text_preview = text_preview[:200] + "..."
        
        return f"""
        <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(59, 130, 246, 0.3);
                    border-radius: 20px; padding: 1.8rem; margin-bottom: 2rem;">
            <h3 style="color: #f8fafc; font-size: 1.2rem; margin-top: 0; margin-bottom: 1rem; font-weight: 700;">
                📝 Content Preview
            </h3>
            <p style="color: #cbd5e1; margin: 0; font-style: italic; line-height: 1.6;">
                "{text_preview}"
            </p>
        </div>
        """
    
    def _format_url_results(self, urls: List[Dict]) -> str:
        """Format URL scan results with UI styling"""
        if not urls:
            return ""
        
        url_html = """
        <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%);
                    backdrop-filter: blur(25px); border-radius: 28px; padding: 2.5rem;
                    margin-bottom: 2rem; border: 2px solid rgba(59, 130, 246, 0.2);
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);">
            <h3 style="color: #f8fafc; font-size: 1.4rem; margin-top: 0; margin-bottom: 1.5rem; font-weight: 700;">
                🔗 Detected URLs
            </h3>
        """
        
        for url_data in urls:
            is_malicious = url_data.get('malicious', 0) > 0
            
            if is_malicious:
                url_bg = "rgba(255, 77, 77, 0.12)"
                url_border = "#ff4d4d"
                url_status = "⚠️ DANGEROUS"
                url_color = "#ff8080"
            else:
                url_bg = "rgba(100, 255, 218, 0.12)"
                url_border = "#64ffda"
                url_status = "✓ SAFE"
                url_color = "#64ffda"
            
            url_html += f"""
            <div style="background: {url_bg}; border: 1px solid {url_border};
                        border-radius: 14px; padding: 1.2rem; margin: 1rem 0;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                    <span style="color: {url_color}; font-weight: 700; font-size: 0.9rem;">
                        {url_status}
                    </span>
                </div>
                <p style="margin: 0.5rem 0; word-break: break-all; color: #cbd5e1; font-size: 0.95rem;">
                    <strong>URL:</strong> {url_data.get('url', 'N/A')}
                </p>
                <p style="margin: 0.8rem 0 0 0; font-size: 0.85rem; color: #94a3b8;">
                    Malicious: <strong>{url_data.get('malicious', 0)}</strong> | 
                    Suspicious: <strong>{url_data.get('suspicious', 0)}</strong> | 
                    Harmless: <strong>{url_data.get('harmless', 0)}</strong>
                </p>
            </div>
            """
        
        url_html += "</div>"
        return url_html

    def send_daily_summary(self, summary_data: Dict) -> bool:
        """
        Send daily summary of threats detected
        
        Args:
            summary_data: Dictionary with daily statistics
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(self.alert_recipients)
            msg['Subject'] = f"📊 PhishGuard Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = self._create_summary_body(summary_data)
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print("✅ Daily summary email sent")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send summary email: {str(e)}")
            return False
    
    def _create_summary_body(self, summary_data: Dict) -> str:
        """Create HTML email body for daily summary with UI styling"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        </head>
        <body style="margin: 0; padding: 40px 20px; font-family: 'Inter', sans-serif;
                    background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #334155 100%);">
            <div style="max-width: 700px; margin: 0 auto;">
                
                <!-- Header -->
                <div style="text-align: center; margin-bottom: 3rem; padding: 3rem 2rem;
                            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
                            border-radius: 32px; border: 2px solid rgba(59, 130, 246, 0.2);
                            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);">
                    <h1 style="font-family: 'Playfair Display', serif; font-size: 3rem; font-weight: 800; margin: 0;
                               background: linear-gradient(135deg, #64ffda 0%, #38bdf8 40%, #2563eb 100%);
                               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        📊 Daily Summary
                    </h1>
                    <p style="font-size: 1.1rem; color: #94a3b8; margin: 1rem 0 0 0;">
                        {today}
                    </p>
                </div>
                
                <!-- Stats Grid -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem;">
                    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(37, 99, 235, 0.15));
                                padding: 2rem; border-radius: 20px; text-align: center;
                                border: 2px solid rgba(59, 130, 246, 0.3);">
                        <h3 style="margin: 0; color: #38bdf8; font-size: 2.5rem; font-weight: 800;">
                            {summary_data.get('total_scans', 0)}
                        </h3>
                        <p style="margin: 0.5rem 0 0 0; color: #cbd5e1; font-weight: 600;">Total Scans</p>
                    </div>
                    <div style="background: linear-gradient(135deg, rgba(255, 77, 77, 0.15), rgba(220, 38, 38, 0.15));
                                padding: 2rem; border-radius: 20px; text-align: center;
                                border: 2px solid rgba(255, 77, 77, 0.3);">
                        <h3 style="margin: 0; color: #ff8080; font-size: 2.5rem; font-weight: 800;">
                            {summary_data.get('threats_detected', 0)}
                        </h3>
                        <p style="margin: 0.5rem 0 0 0; color: #cbd5e1; font-weight: 600;">Threats Detected</p>
                    </div>
                </div>
                
                <!-- Breakdown -->
                <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%);
                            backdrop-filter: blur(25px); border-radius: 28px; padding: 2.5rem;
                            border: 2px solid rgba(59, 130, 246, 0.2);
                            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);">
                    <h3 style="color: #f8fafc; font-size: 1.4rem; margin-top: 0; margin-bottom: 1.5rem;">
                        Threat Breakdown
                    </h3>
                    
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem 1.5rem; margin: 0.8rem 0;
                                border-radius: 14px; border: 1px solid rgba(59, 130, 246, 0.2);">
                        <span style="color: #cbd5e1; font-weight: 600;">Text Phishing:</span>
                        <span style="color: #e2e8f0; font-weight: 700; float: right;">
                            {summary_data.get('text_phishing', 0)}
                        </span>
                    </div>
                    
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem 1.5rem; margin: 0.8rem 0;
                                border-radius: 14px; border: 1px solid rgba(59, 130, 246, 0.2);">
                        <span style="color: #cbd5e1; font-weight: 600;">Image Phishing:</span>
                        <span style="color: #e2e8f0; font-weight: 700; float: right;">
                            {summary_data.get('image_phishing', 0)}
                        </span>
                    </div>
                    
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem 1.5rem; margin: 0.8rem 0;
                                border-radius: 14px; border: 1px solid rgba(59, 130, 246, 0.2);">
                        <span style="color: #cbd5e1; font-weight: 600;">Malicious URLs:</span>
                        <span style="color: #e2e8f0; font-weight: 700; float: right;">
                            {summary_data.get('malicious_urls', 0)}
                        </span>
                    </div>
                </div>
                
                <!-- Footer -->
                <div style="text-align: center; margin-top: 3rem; padding-top: 2rem;
                            border-top: 2px solid rgba(59, 130, 246, 0.2); color: #94a3b8; font-size: 0.9rem;">
                    <p style="margin: 0.5rem 0;">PhishGuard AI - Advanced Threat Detection</p>
                    <p style="margin: 1rem 0 0 0; font-size: 0.8rem; color: #64748b;">
                        © 2025 PhishGuard AI. All rights reserved.
                    </p>
                </div>
                
            </div>
        </body>
        </html>
        """
        return html


# Usage example
if __name__ == "__main__":
    alerter = EmailAlerter()
    
    # Example threat data
    threat = {
        "label": "phishing",
        "confidence": 0.95,
        "content_type": "Text + URL",
        "text_preview": "Urgent: Your account will be suspended unless you verify your credentials immediately...",
        "urls": [
            {"url": "http://suspicious-site.com", "malicious": 15, "suspicious": 3, "harmless": 2}
        ]
    }
    
    alerter.send_threat_alert(threat, severity="HIGH")