import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from dotenv import load_dotenv
from portkey_ai import Portkey
import httpx
import time

load_dotenv()

@st.cache_resource
def get_portkey_client():
    custom_timeout = httpx.Timeout(30.0, connect=30.0)
    # Initialize Portkey client exactly as specified
    client = Portkey(
        api_key=st.secrets["PORTKEY_API_KEY"],
        base_url="https://eu.aigw.galileo.roche.com/v1",
        debug=False,
        provider='azure-openai',
        timeout=custom_timeout,
    )
    return client

def call_llm_with_retries(portkey, event_type, context, retries=3, wait=5):
    prompt = f"""
Type of Event: {event_type}
Context Information:
{context}
Please provide a risk score between 0 and 1 and a rationale.
Format:
Score: <number>
Rationale: <text>
"""
    messages = [{"role": "user", "content": prompt}]
    attempt = 0
    while attempt < retries:
        try:
            completion = portkey.chat.completions.create(
                messages=messages,
                model="gpt-4o-2024-11-20",
                max_tokens=300,
                temperature=0.6,
                stream=False,
                request_timeout=20000
            )
            output_text = completion.choices[0].message.content.strip()
            score, rationale = None, None
            for line in output_text.splitlines():
                if line.lower().startswith("score:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                    except:
                        pass
                elif line.lower().startswith("rationale:"):
                    rationale = line.split(":", 1)[1].strip()
            return score, rationale
        except Exception as e:
            st.warning(f"API call failed on attempt {attempt+1}: {e}")
            time.sleep(wait)
            attempt += 1
    return None, None

def fallback_risk(event_type, event):
    # Same heuristic fallback logic as before
    risk = 0.0
    reasons = []
    if event_type == "Clinical Trial":
        status = str(event.get('OverallStatus', '')).upper()
        phase = str(event.get('Phase', '')).upper()
        enrollment = event.get('EnrollmentCount', 0)
        if status in ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']:
            risk += 0.7
            reasons.append(f"Trial status '{status}'")
        if 'PHASE 1' in phase:
            risk += 0.5
            reasons.append("Early phase 1 trial")
        elif 'PHASE 2' in phase:
            risk += 0.3
            reasons.append("Phase 2 trial")
        if isinstance(enrollment, (int, float)):
            if enrollment < 30:
                risk += 0.25
                reasons.append(f"Very low enrollment ({enrollment})")
            elif enrollment < 100:
                risk += 0.1
                reasons.append(f"Low enrollment ({enrollment})")
            else:
                risk -= 0.05
                reasons.append("Healthy enrollment")
        rationale = "Fallback heuristic due to missing LLM score: " + "; ".join(reasons)
        return min(max(risk, 0), 1), rationale
    elif event_type == "Press Release":
        summary = event.get('summary', '').lower()
        risk = 0.2 if summary else 0.1
        reasons.append("Press release text available")
        keywords = ["warning", "safety", "risk", "terminated", "adverse", "recall"]
        if any(k in summary for k in keywords):
            risk = max(risk, 0.5)
            reasons.append("Risk keywords present")
        rationale = "Fallback heuristic due to missing LLM score: " + "; ".join(reasons)
        return min(max(risk, 0), 1), rationale
    return 0.1, "Fallback heuristic applied"

def build_timeline(portkey, trials_df, press_df):
    trials_df = trials_df.sort_values('StartDate')
    timeline_events = []
    for idx, trial in trials_df.iterrows():
        context = f"""
Trial ID: {trial.get('NCTId')}
Phase: {trial.get('Phase', '')}
Status: {trial.get('OverallStatus', '')}
Enrollment: {trial.get('EnrollmentCount', '')}
Summary: {trial.get('BriefSummary', '')}
"""
        past = "\n".join(trials_df.iloc[max(0, idx-3):idx]['BriefSummary'].fillna('No summary'))
        context += f"\nPast trials:\n{past}"
        score, rationale = call_llm_with_retries(portkey, "Clinical Trial", context)
        if score is None:
            score, rationale = fallback_risk("Clinical Trial", trial)
        timeline_events.append({
            "Date": trial.get('StartDate'),
            "EventType": "ClinicalTrial",
            "Title": trial.get('BriefTitle', ''),
            "RiskScore": score,
            "RiskExplanation": rationale or "",
        })
    for _, press in press_df.sort_values('Date').iterrows():
        context = f"""
Headline: {press.get('headline','')}
Summary: {press.get('summary','')}
Source: {press.get('source','')}
Related: {press.get('related','')}
"""
        score, rationale = call_llm_with_retries(portkey, "Press Release", context)
        if score is None:
            score, rationale = fallback_risk("Press Release", press)
        timeline_events.append({
            "Date": press.get('Date'),
            "EventType": "PressRelease",
            "Title": press.get('headline', ''),
            "RiskScore": score,
            "RiskExplanation": rationale or "",
        })
    df = pd.DataFrame(timeline_events)
    df.dropna(subset=['Date'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['RiskScore'] = pd.to_numeric(df['RiskScore'], errors='coerce')
    df.sort_values('Date', inplace=True)
    return df

def main():
    st.title("Zilebesiran Clinical Trial Risk Dashboard")

    portkey = get_portkey_client()

    uploaded_trials = st.file_uploader("Upload Clinical Trials CSV", type=["csv"])
    uploaded_press = st.file_uploader("Upload Press Releases CSV", type=["csv"])

    if uploaded_trials is None or uploaded_press is None:
        st.info("Please upload clinical trials and press releases CSV files to start.")
        return

    trials_df = pd.read_csv(uploaded_trials, parse_dates=['StartDate', 'CompletionDate'])
    press_df = pd.read_csv(uploaded_press)
    press_df['Date'] = pd.to_datetime(press_df['datetime'], unit='s', errors='coerce')

    timeline_df = build_timeline(portkey, trials_df, press_df)

    st.subheader("Risk Monitoring Table")

    def risk_color(val):
        if pd.isnull(val):
            return ''
        if val > 0.7:
            return 'background-color: #fa8072; color: white; font-weight: bold;'
        elif val > 0.4:
            return 'background-color: #ffc107; font-weight: bold;'
        return ''

    st.dataframe(timeline_df.style.applymap(risk_color, subset=['RiskScore']))

    st.subheader("Risk Score Timeline")

    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = {'ClinicalTrial': '#1f77b4', 'PressRelease': '#ff7f0e'}
    markers = {'ClinicalTrial': 'o', 'PressRelease': 's'}
    for etype in timeline_df['EventType'].unique():
        subset = timeline_df[timeline_df['EventType'] == etype]
        ax.plot(subset['Date'], subset['RiskScore'], label=etype, color=colors[etype], marker=markers[etype], linestyle='-')
        for _, row in subset.iterrows():
            label = row['Title'] if len(row['Title']) <= 30 else row['Title'][:27] + "..."
            ax.annotate(label, (row['Date'], row['RiskScore']), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8, color=colors[etype])
    ax.set_ylim(0, 1.1)
    ax.set_title("Risk Score Timeline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Risk Score")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

if __name__ == "__main__":
    main()

