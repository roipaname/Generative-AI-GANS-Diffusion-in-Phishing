# url_scanner.py
import requests
import os
import streamlit as st
import time

VT_API_KEY = "b293d21cec19d6b735165ba86bb15f9e6bc7eb1a86976221c32930d1599c33cb"
VT_BASE = "https://www.virustotal.com/api/v3"


def scan_url(url: str):
    """Submit a URL for scanning to VirusTotal."""
    headers = {"x-apikey": VT_API_KEY}
    try:
        resp = requests.post(f"{VT_BASE}/urls", data={"url": url}, headers=headers)
        if resp.ok:
            analysis_id = resp.json().get("data", {}).get("id")
            if analysis_id:
                return analysis_id
            else:
                #st.error("⚠️ Could not retrieve analysis ID from response.")
                # Automatically mark as malicious if response structure invalid
                return "AUTO_MALICIOUS"
        else:
            #st.error(f"🚨 Failed to submit URL ({resp.status_code}): {resp.text}")
            # Automatically mark as malicious on failed submission
            return "AUTO_MALICIOUS"
    except Exception as e:
        #st.error(f"🚨 Network error submitting URL: {e}")
        # Automatically mark as malicious on exception
        return "AUTO_MALICIOUS"


def get_analysis_result(analysis_id: str):
    """Poll the VirusTotal API until scan results are ready."""
    if analysis_id == "AUTO_MALICIOUS":
        return {"auto_flagged": True}

    headers = {"x-apikey": VT_API_KEY}
    analysis_url = f"{VT_BASE}/analyses/{analysis_id}"

    for _ in range(10):  # Poll up to 10 times
        try:
            resp = requests.get(analysis_url, headers=headers)
            if not resp.ok:
                #st.error("⚠️ Error fetching VirusTotal results.")
                return {"auto_flagged": True}

            data = resp.json().get("data", {})
            attributes = data.get("attributes", {})
            status = attributes.get("status")

            if status == "completed":
                return data

            time.sleep(2)

        except Exception as e:
            #st.error(f"⚠️ Error while polling results: {e}")
            return {"auto_flagged": True}

    st.warning("⏱️ Timed out waiting for VirusTotal results.")
    # Automatically flag as malicious on timeout
    return {"auto_flagged": True}


def interpret_results(data: dict):
    """Summarize the scan result for the UI."""
    # Handle auto-malicious case
    if data.get("auto_flagged"):
        return {
            "verdict": "Phishing / Malicious (Auto-Flagged)",
            "malicious": 1,
            "suspicious": 0,
            "harmless": 0,
            "undetected": 0,
        }

    stats = data.get("attributes", {}).get("stats", {})
    harmless = stats.get("harmless", 0)
    malicious = stats.get("malicious", 0)
    suspicious = stats.get("suspicious", 0)
    undetected = stats.get("undetected", 0)

    verdict = "Legitimate"
    if suspicious > 1:
        verdict = "Phishing / Suspicious"
    if malicious > 0 :
        verdict = "Phishing / Malicious"
  

    return {
        "verdict": verdict,
        "malicious": malicious,
        "suspicious": suspicious,
        "harmless": harmless,
        "undetected": undetected,
    }
