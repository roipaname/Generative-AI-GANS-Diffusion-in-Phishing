import streamlit as st
import json
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pandas as pd
from collections import Counter


logo_path = "./frontend/images/PhishGuard_logo.png"  
# Page config
st.set_page_config(
    page_title="PhishGuard Security Dashboard",
    page_icon=logo_path,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("./frontend/style.css")

# Custom CSS for smooth, minimalistic design
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(100, 255, 218, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(100, 255, 218, 0.3);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #64ffda 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .metric-subtitle {
        font-size: 0.85rem;
        color: #64ffda;
        margin-top: 0.5rem;
    }
    
    .danger-value {
        background: linear-gradient(135deg, #ff4d4d 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .success-value {
        background: linear-gradient(135deg, #64ffda 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .chart-container {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(100, 255, 218, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .section-header {
        font-size: 1.3rem;
        color: #f8fafc;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-critical { background: rgba(255, 77, 77, 0.2); color: #ff4d4d; }
    .badge-high { background: rgba(251, 191, 36, 0.2); color: #fbbf24; }
    .badge-medium { background: rgba(251, 146, 60, 0.2); color: #fb923c; }
    .badge-low { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }
    .badge-safe { background: rgba(100, 255, 218, 0.2); color: #64ffda; }
</style>
""", unsafe_allow_html=True)

# Data directory
PROCESSED_DATA_DIR = Path("./processed_data")
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

def load_all_scans():
    """Load all scan data from processed_data directory"""
    scans = []
    if PROCESSED_DATA_DIR.exists():
        for file in sorted(PROCESSED_DATA_DIR.glob("*.json"), reverse=True):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    data['filename'] = file.name
                    scans.append(data)
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
    return scans

def get_enhanced_metrics(scans):
    """Calculate enhanced threat metrics from scan data"""
    total_scans = len(scans)
    if total_scans == 0:
        return None
    
    phishing_detected = sum(1 for s in scans if s.get('is_phishing', False))
    legitimate = total_scans - phishing_detected
    
    # Risk levels
    critical = sum(1 for s in scans if s.get('risk_level') == 'critical')
    high_risk = sum(1 for s in scans if s.get('risk_level') == 'high')
    medium_risk = sum(1 for s in scans if s.get('risk_level') == 'medium')
    low_risk = sum(1 for s in scans if s.get('risk_level') == 'low')
    
    # Scan types
    text_scans = sum(1 for s in scans if s.get('scan_type') == 'text')
    image_scans = sum(1 for s in scans if s.get('scan_type') == 'image')
    combined_scans = sum(1 for s in scans if s.get('scan_type') == 'combined')
    voice_scans = sum(1 for s in scans if s.get('scan_type') == 'voice')
    
    # URL analysis
    total_urls = sum(s.get('urls_detected', 0) for s in scans)
    total_malicious = sum(s.get('malicious_urls', 0) for s in scans)
    total_suspicious = sum(s.get('suspicious_urls', 0) for s in scans)
    
    # Confidence analysis
    confidences = [s.get('confidence', 0) for s in scans if 'confidence' in s]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    max_confidence = max(confidences) if confidences else 0
    min_confidence = min(confidences) if confidences else 0
    
    # Time-based metrics
    now = datetime.now()
    last_24h = sum(1 for s in scans if datetime.fromisoformat(s.get('timestamp', '2000-01-01')) > now - timedelta(hours=24))
    last_7d = sum(1 for s in scans if datetime.fromisoformat(s.get('timestamp', '2000-01-01')) > now - timedelta(days=7))
    
    threats_24h = sum(1 for s in scans if s.get('is_phishing') and datetime.fromisoformat(s.get('timestamp', '2000-01-01')) > now - timedelta(hours=24))
    threats_7d = sum(1 for s in scans if s.get('is_phishing') and datetime.fromisoformat(s.get('timestamp', '2000-01-01')) > now - timedelta(days=7))
    
    # Detection rate
    detection_rate = (phishing_detected / total_scans * 100) if total_scans > 0 else 0
    
    return {
        'total_scans': total_scans,
        'phishing_detected': phishing_detected,
        'legitimate': legitimate,
        'critical': critical,
        'high_risk': high_risk,
        'medium_risk': medium_risk,
        'low_risk': low_risk,
        'text_scans': text_scans,
        'image_scans': image_scans,
        'combined_scans': combined_scans,
        'voice_scans': voice_scans,
        'total_urls': total_urls,
        'malicious_urls': total_malicious,
        'suspicious_urls': total_suspicious,
        'avg_confidence': avg_confidence,
        'max_confidence': max_confidence,
        'min_confidence': min_confidence,
        'scans_24h': last_24h,
        'scans_7d': last_7d,
        'threats_24h': threats_24h,
        'threats_7d': threats_7d,
        'detection_rate': detection_rate
    }

def create_radial_gauge(value, title, color_start, color_end):
    """Create a modern radial gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#e2e8f0'}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#f8fafc'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 0, 'tickcolor': "rgba(0,0,0,0)"},
            'bar': {'color': f'rgba({color_start}, 0.8)', 'thickness': 0.7},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 100], 'color': 'rgba(255,255,255,0.05)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e2e8f0"},
        height=200,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_threat_timeline(scans):
    """Create an enhanced timeline with hourly breakdown"""
    df_data = []
    for scan in scans:
        timestamp = datetime.fromisoformat(scan.get('timestamp', datetime.now().isoformat()))
        df_data.append({
            'timestamp': timestamp,
            'is_phishing': scan.get('is_phishing', False),
            'confidence': scan.get('confidence', 0),
            'risk_level': scan.get('risk_level', 'unknown')
        })
    
    if not df_data:
        return None
        
    df = pd.DataFrame(df_data)
    df['hour'] = df['timestamp'].dt.floor('H')
    
    hourly = df.groupby('hour').agg({
        'is_phishing': ['sum', 'count']
    }).reset_index()
    hourly.columns = ['hour', 'threats', 'total']
    hourly['safe'] = hourly['total'] - hourly['threats']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly['hour'],
        y=hourly['threats'],
        name='Threats',
        mode='lines+markers',
        line=dict(color='#ff4d4d', width=3),
        marker=dict(size=8, symbol='circle'),
        fill='tozeroy',
        fillcolor='rgba(255, 77, 77, 0.2)',
        hovertemplate='<b>Threats</b>: %{y}<br>%{x}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly['hour'],
        y=hourly['safe'],
        name='Safe',
        mode='lines+markers',
        line=dict(color='#64ffda', width=3),
        marker=dict(size=8, symbol='circle'),
        fill='tozeroy',
        fillcolor='rgba(100, 255, 218, 0.2)',
        hovertemplate='<b>Safe</b>: %{y}<br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Detection Timeline",
        xaxis_title="Time",
        yaxis_title="Scans",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        font={'color': "#e2e8f0"},
        height=300,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    
    return fig

def create_risk_heatmap(scans):
    """Create a risk level distribution heatmap"""
    risk_counts = Counter(s.get('risk_level', 'unknown') for s in scans)
    
    labels = ['Critical', 'High', 'Medium', 'Low', 'Safe']
    values = [
        risk_counts.get('critical', 0),
        risk_counts.get('high', 0),
        risk_counts.get('medium', 0),
        risk_counts.get('low', 0),
        sum(1 for s in scans if not s.get('is_phishing', False))
    ]
    colors = ['#dc2626', '#ff4d4d', '#fbbf24', '#fb923c', '#64ffda']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                line=dict(color='rgba(30, 41, 59, 0.5)', width=1)
            ),
            text=values,
            textposition='outside',
            textfont=dict(size=18, color='#f8fafc', family='monospace'),
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Risk Distribution",
        xaxis_title="",
        yaxis_title="Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        font={'color': "#e2e8f0"},
        height=300,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    
    return fig

def create_scan_type_donut(metrics):
    """Create a modern donut chart for scan types"""
    labels = ['Text', 'Image', 'Combined', 'Voice']
    values = [metrics['text_scans'], metrics['image_scans'], 
              metrics['combined_scans'], metrics['voice_scans']]
    
    colors = ['#64ffda', '#38bdf8', '#2563eb', '#a78bfa']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color='#1e293b', width=2)),
        textfont=dict(size=13, color='#f8fafc', family='monospace'),
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>%{value} scans<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Scan Types",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e2e8f0"},
        height=300,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1),
        margin=dict(l=20, r=100, t=60, b=20)
    )
    
    fig.add_annotation(
        text=f"<b>{sum(values)}</b><br>Total",
        x=0.5, y=0.5,
        font=dict(size=20, color='#f8fafc'),
        showarrow=False
    )
    
    return fig

def create_confidence_distribution(scans):
    """Create confidence score distribution"""
    phishing_scans = [s for s in scans if s.get('is_phishing')]
    if not phishing_scans:
        return None
    
    confidences = [s.get('confidence', 0) * 100 for s in phishing_scans]
    
    fig = go.Figure(data=[go.Histogram(
        x=confidences,
        nbinsx=20,
        marker=dict(
            color='#38bdf8',
            line=dict(color='#1e293b', width=1)
        ),
        hovertemplate='Confidence: %{x:.1f}%<br>Count: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Threat Confidence Distribution",
        xaxis_title="Confidence Score (%)",
        yaxis_title="Frequency",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        font={'color': "#e2e8f0"},
        height=300,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    
    return fig

def create_hourly_heatmap(scans):
    """Create hourly activity heatmap"""
    df_data = []
    for scan in scans:
        timestamp = datetime.fromisoformat(scan.get('timestamp', datetime.now().isoformat()))
        df_data.append({
            'hour': timestamp.hour,
            'day': timestamp.strftime('%A'),
            'is_phishing': 1 if scan.get('is_phishing') else 0
        })
    
    if not df_data:
        return None
    
    df = pd.DataFrame(df_data)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    pivot = df.pivot_table(
        values='is_phishing',
        index='day',
        columns='hour',
        aggfunc='sum',
        fill_value=0
    )
    
    pivot = pivot.reindex(days_order, fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f'{h:02d}:00' for h in pivot.columns],
        y=pivot.index,
        colorscale=[
            [0, 'rgba(100, 255, 218, 0.1)'],
            [0.3, 'rgba(56, 189, 248, 0.4)'],
            [0.6, 'rgba(251, 191, 36, 0.6)'],
            [1, 'rgba(255, 77, 77, 0.9)']
        ],
        hovertemplate='%{y}<br>%{x}<br>Threats: %{z}<extra></extra>',
        colorbar=dict(
            title=dict(text='Threats', side='right'),
            tickmode='linear'
        )
    ))
    
    fig.update_layout(
        title="Threat Activity Heatmap",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        font={'color': "#e2e8f0"},
        height=350,
        margin=dict(l=80, r=80, t=60, b=40)
    )
    
    return fig

def create_verdict_breakdown(scans):
    """Create overall verdict breakdown"""
    verdicts = Counter(s.get('overall_verdict', 'unknown') for s in scans)
    
    labels = list(verdicts.keys())
    values = list(verdicts.values())
    
    colors_map = {
        'phishing': '#ff4d4d',
        'legitimate': '#64ffda',
        'suspicious': '#fbbf24',
        'unknown': '#94a3b8'
    }
    colors = [colors_map.get(label, '#94a3b8') for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=[l.title() for l in labels],
        values=values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color='#1e293b', width=2)),
        textfont=dict(size=14, color='#f8fafc'),
        textposition='outside',
        hovertemplate='<b>%{label}</b><br>%{value} scans<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Overall Verdict Distribution",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e2e8f0"},
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=60, b=60)
    )
    
    return fig

def create_text_length_analysis(scans):
    """Analyze text length distribution"""
    text_scans = [s for s in scans if s.get('text_length', 0) > 0]
    if not text_scans:
        return None
    
    phishing_lengths = [s.get('text_length', 0) for s in text_scans if s.get('is_phishing')]
    legitimate_lengths = [s.get('text_length', 0) for s in text_scans if not s.get('is_phishing')]
    
    fig = go.Figure()
    
    if phishing_lengths:
        fig.add_trace(go.Box(
            y=phishing_lengths,
            name='Phishing',
            marker=dict(color='#ff4d4d'),
            boxmean='sd',
            hovertemplate='Length: %{y}<extra></extra>'
        ))
    
    if legitimate_lengths:
        fig.add_trace(go.Box(
            y=legitimate_lengths,
            name='Legitimate',
            marker=dict(color='#64ffda'),
            boxmean='sd',
            hovertemplate='Length: %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Text Length Analysis",
        yaxis_title="Character Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        font={'color': "#e2e8f0"},
        height=300,
        showlegend=True,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    
    return fig

def create_url_threat_gauge(metrics):
    """Create gauge for URL threat percentage"""
    if metrics['total_urls'] == 0:
        threat_pct = 0
    else:
        threat_pct = ((metrics['malicious_urls'] + metrics['suspicious_urls']) / metrics['total_urls']) * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=threat_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "URL Threat Rate", 'font': {'size': 18, 'color': '#e2e8f0'}},
        number={'suffix': "%", 'font': {'size': 36, 'color': '#f8fafc'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#64ffda"},
            'bar': {'color': "#ff4d4d", 'thickness': 0.75},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 2,
            'bordercolor': "rgba(255, 77, 77, 0.3)",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(100, 255, 218, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(251, 191, 36, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(251, 146, 60, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(255, 77, 77, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e2e8f0"},
        height=280,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_daily_trend(scans):
    """Create daily trend comparison"""
    df_data = []
    for scan in scans:
        timestamp = datetime.fromisoformat(scan.get('timestamp', datetime.now().isoformat()))
        df_data.append({
            'date': timestamp.date(),
            'is_phishing': scan.get('is_phishing', False)
        })
    
    if not df_data:
        return None
    
    df = pd.DataFrame(df_data)
    daily = df.groupby('date').agg({
        'is_phishing': ['sum', 'count']
    }).reset_index()
    daily.columns = ['date', 'threats', 'total']
    daily['legitimate'] = daily['total'] - daily['threats']
    daily['threat_rate'] = (daily['threats'] / daily['total'] * 100).round(1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily['date'],
        y=daily['threats'],
        name='Threats',
        marker=dict(color='#ff4d4d'),
        hovertemplate='<b>Threats</b>: %{y}<br>%{x}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=daily['date'],
        y=daily['legitimate'],
        name='Legitimate',
        marker=dict(color='#64ffda'),
        hovertemplate='<b>Legitimate</b>: %{y}<br>%{x}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['threat_rate'],
        name='Threat Rate %',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='#fbbf24', width=3),
        marker=dict(size=8, symbol='diamond'),
        hovertemplate='<b>Threat Rate</b>: %{y:.1f}%<br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Daily Scan Trends",
        xaxis_title="Date",
        yaxis_title="Scan Count",
        yaxis2=dict(
            title='Threat Rate (%)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.03)',
        font={'color': "#e2e8f0"},
        height=350,
        barmode='stack',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=60, t=60, b=40)
    )
    
    return fig

# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0 3rem 0;'>
    <h1 style='font-size: 3rem; font-weight: 700; margin: 0;
               background: linear-gradient(135deg, #64ffda 0%, #38bdf8 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🛡️ PhishGuard
    </h1>
    <p style='color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem;'>
        Advanced Threat Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
scans = load_all_scans()
metrics = get_enhanced_metrics(scans)

if metrics is None:
    st.info("📊 No scan data available yet. Start analyzing content to see results here!")
else:
    # Primary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Scans</div>
            <div class='metric-value'>{metrics['total_scans']:,}</div>
            <div class='metric-subtitle'>↑ {metrics['scans_24h']} in last 24h</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Threats Detected</div>
            <div class='metric-value danger-value'>{metrics['phishing_detected']:,}</div>
            <div class='metric-subtitle'>⚠️ {metrics['threats_24h']} in last 24h</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Detection Rate</div>
            <div class='metric-value danger-value'>{metrics['detection_rate']:.1f}%</div>
            <div class='metric-subtitle'>of all scans</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Avg Confidence</div>
            <div class='metric-value'>{metrics['avg_confidence']:.1%}</div>
            <div class='metric-subtitle'>Max: {metrics['max_confidence']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Secondary Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card' style='padding: 1rem;'>
            <div class='metric-label' style='font-size: 0.75rem;'>Critical</div>
            <div class='metric-value' style='font-size: 1.8rem;'>{metrics['critical']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card' style='padding: 1rem;'>
            <div class='metric-label' style='font-size: 0.75rem;'>High Risk</div>
            <div class='metric-value' style='font-size: 1.8rem;'>{metrics['high_risk']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card' style='padding: 1rem;'>
            <div class='metric-label' style='font-size: 0.75rem;'>Medium</div>
            <div class='metric-value' style='font-size: 1.8rem;'>{metrics['medium_risk']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card' style='padding: 1rem;'>
            <div class='metric-label' style='font-size: 0.75rem;'>URLs Scanned</div>
            <div class='metric-value' style='font-size: 1.8rem;'>{metrics['total_urls']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class='metric-card' style='padding: 1rem;'>
            <div class='metric-label' style='font-size: 0.75rem;'>Malicious URLs</div>
            <div class='metric-value danger-value' style='font-size: 1.8rem;'>{metrics['malicious_urls']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Section
    st.markdown("<div class='section-header'>📊 Analytics Overview</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        timeline = create_threat_timeline(scans)
        if timeline:
            st.plotly_chart(timeline, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        risk_chart = create_risk_heatmap(scans)
        st.plotly_chart(risk_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Daily Trends
    st.markdown("<div class='section-header'>📈 Trend Analysis</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    daily_trend = create_daily_trend(scans)
    if daily_trend:
        st.plotly_chart(daily_trend, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # URL & Scan Type Analysis
    st.markdown("<div class='section-header'>🔍 Detection Insights</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        scan_type_chart = create_scan_type_donut(metrics)
        st.plotly_chart(scan_type_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        verdict_chart = create_verdict_breakdown(scans)
        st.plotly_chart(verdict_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        url_gauge = create_url_threat_gauge(metrics)
        st.plotly_chart(url_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Advanced Analytics
    st.markdown("<div class='section-header'>🧬 Advanced Analytics</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        confidence_chart = create_confidence_distribution(scans)
        if confidence_chart:
            st.plotly_chart(confidence_chart, use_container_width=True)
        else:
            st.info("No threat confidence data available")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        text_analysis = create_text_length_analysis(scans)
        if text_analysis:
            st.plotly_chart(text_analysis, use_container_width=True)
        else:
            st.info("No text length data available")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Heatmap
    st.markdown("<div class='section-header'>🗓️ Temporal Patterns</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    heatmap = create_hourly_heatmap(scans)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Insufficient data for heatmap analysis")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Recent Activity
    st.markdown("<div class='section-header'>🕐 Recent Activity</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    
    table_data = []
    for scan in scans[:30]:
        timestamp = datetime.fromisoformat(scan.get('timestamp', datetime.now().isoformat()))
        
        risk = scan.get('risk_level', 'unknown')
        if risk == 'critical':
            badge = '<span class="status-badge badge-critical">CRITICAL</span>'
        elif risk == 'high':
            badge = '<span class="status-badge badge-high">HIGH</span>'
        elif risk == 'medium':
            badge = '<span class="status-badge badge-medium">MEDIUM</span>'
        elif risk == 'low':
            badge = '<span class="status-badge badge-low">LOW</span>'
        else:
            badge = '<span class="status-badge badge-safe">SAFE</span>'
        
        result = '🚨 Phishing' if scan.get('is_phishing') else '✅ Safe'
        
        table_data.append({
            'Time': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Type': scan.get('scan_type', 'unknown').title(),
            'Result': result,
            'Risk': risk.title(),
            'Confidence': f"{scan.get('confidence', 0):.1%}",
            'No URLs': scan.get('urls_detected', 0),
            "URLS": ",".join(scan.get("urls", [])),
            "Content":scan.get("text_content","")
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style='text-align: center; margin-top: 3rem; padding: 2rem; 
            color: rgba(255,255,255,0.4); border-top: 1px solid rgba(100, 255, 218, 0.1);'>
    <p style='font-size: 0.85rem;'>🛡️ PhishGuard Security Platform — Real-time Threat Intelligence</p>
    <p style='font-size: 0.75rem; margin-top: 0.5rem;'>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)