import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from dotenv import load_dotenv
from portkey_ai import Portkey
import httpx

# Load env and init Portkey client (same as your previous code)
load_dotenv()
custom_timeout = httpx.Timeout(30.0, connect=30.0)
portkey = Portkey(
    api_key="eIp6VbA0BucuOCUr3Z0lw7UJ9Ls/",
    base_url="https://eu.aigw.galileo.roche.com/v1",
    debug=False,
    provider='azure-openai',
    timeout=custom_timeout
)

def base_heuristic_risk(event_type, event):
    # (Same heuristic function as before)
    risk = 0.0
    reasons = []
    if event_type == "Clinical Trial":
        status = str(event.get('OverallStatus', '')).upper()
        phase = str(event.get('Phase', '')).upper()
        enrollment = event.get('EnrollmentCount', 0)
        if status in ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']:
            risk += 0.7
            reasons.append(f"Trial status '{status}' indicates elevated risk")
        if 'PHASE 1' in phase:
            risk += 0.5
            reasons.append("Early phase 1 trial")
        elif 'PHASE 2' in phase:
            risk += 0.3
            reasons.append("Phase 2 trial")
        if isinstance(enrollment, (int,float)):
            if enrollment < 30:
                risk += 0.25
                reasons.append(f"Very low enrollment ({enrollment})")
            elif enrollment < 100:
                risk += 0.1
                reasons.append(f"Low enrollment ({enrollment})")
            else:
                risk -= 0.05
                reasons.append("Healthy enrollment levels")

        rationale = "Fallback heuristic risk score applied due to missing LLM score. " + "; ".join(reasons)
        return min(max(risk, 0), 1), rationale

    elif event_type == "Press Release":
        summary = event.get('summary', '').lower()
        risk = 0.2 if summary else 0.1
        reasons.append("Press release textual information available")
        keywords = ["warning", "safety", "risk", "terminated", "adverse", "recall"]
        if any(k in summary for k in keywords):
            risk = max(risk, 0.5)
            reasons.append("Risk-related keywords found in press release")

        rationale = "Fallback heuristic risk score applied due to missing LLM score. " + "; ".join(reasons)
        return min(max(risk, 0), 1), rationale

    return 0.1, "Fallback heuristic risk score applied due to insufficient information."

def call_llm_for_risk(event_type, context_blocks, previous_risk=None):
    input_text = f"""
Type of Event: {event_type}
Context Information:
{context_blocks}
Previous Risk Score: {previous_risk}
As a clinical data monitoring expert, output a refined risk score between 0 and 1 for this event, and a brief rationale.
Format:
Score: <number>
Rationale: <text>
"""
    messages = [{"role": "user", "content": input_text}]
    completion = portkey.chat.completions.create(
        messages=messages,
        model="gpt-4o-2024-11-20",
        max_tokens=300,
        temperature=0.6,
        stream=False
    )
    output_text = completion.choices[0].message.content.strip()
    score, rationale = None, None
    for line in output_text.splitlines():
        lower_line = line.lower()
        if lower_line.startswith("score:"):
            try:
                score = float(line.split(":",1)[1].strip())
            except:
                pass
        elif lower_line.startswith("rationale:"):
            rationale = line.split(":",1)[1].strip()
    return score, rationale

def build_timeline(trials_df, press_df, n_past=3):
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
        past_summaries = "\n".join(trials_df.iloc[max(0, idx-n_past):idx]['BriefSummary'].fillna('No summary').tolist())
        context += f"\nRecent Past Clinical Trials Summaries:\n{past_summaries}"
        score, rationale = call_llm_for_risk("Clinical Trial", context)
        if score is None:
            score, rationale = base_heuristic_risk("Clinical Trial", trial)
        timeline_events.append({
            "Date": trial.get('StartDate'),
            "EventType": "ClinicalTrial",
            "Title": trial.get('BriefTitle', ''),
            "RiskScore": score,
            "RiskExplanation": rationale or '',
        })
    for _, press in press_df.sort_values('Date').iterrows():
        context = f"""
Headline: {press.get('headline','')}
Summary: {press.get('summary','')}
Source: {press.get('source','')}
Related: {press.get('related','')}
"""
        score, rationale = call_llm_for_risk("Press Release", context)
        if score is None:
            score, rationale = base_heuristic_risk("Press Release", press)
        timeline_events.append({
            "Date": press.get('Date'),
            "EventType": "PressRelease",
            "Title": press.get('headline', ''),
            "RiskScore": score,
            "RiskExplanation": rationale or '',
        })
    timeline_df = pd.DataFrame(timeline_events)
    timeline_df.dropna(subset=['Date'], inplace=True)
    timeline_df['Date'] = pd.to_datetime(timeline_df['Date'])
    timeline_df['RiskScore'] = pd.to_numeric(timeline_df['RiskScore'], errors='coerce')
    timeline_df.sort_values('Date', inplace=True)
    return timeline_df

def main():
    st.title("Zilebesiran Clinical Trial Risk Monitoring")

    trials_df = pd.read_csv("zilebesiran_clinical_trials.csv", parse_dates=['StartDate', 'CompletionDate'])
    press_df = pd.read_csv("zilebesiran_filtered_news.csv")
    press_df['Date'] = pd.to_datetime(press_df['datetime'], unit='s', errors='coerce')

    st.sidebar.header("Filters")
    event_filter = st.sidebar.multiselect("Select Event Types", options=["ClinicalTrial", "PressRelease"], default=["ClinicalTrial","PressRelease"])

    timeline_df = build_timeline(trials_df, press_df)

    # Apply filter
    plot_df = timeline_df[timeline_df.EventType.isin(event_filter)]

    # Show risk table
    st.subheader("Risk Monitoring Table")
    def color_risk(val):
        if pd.isnull(val): 
            return ''
        if val > 0.7: return 'background-color: #ff4d4d; color: white; font-weight:bold;'
        elif val > 0.4: return 'background-color: #ffcc00; font-weight:bold;'
        else: return ''
    st.dataframe(plot_df.style.applymap(color_risk, subset=['RiskScore']))

    # Plot the timeline
    st.subheader("Risk Score Timeline")
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(15,7))
    colors = {'ClinicalTrial': '#1f77b4', 'PressRelease': '#ff7f0e'}
    markers = {'ClinicalTrial': 'o', 'PressRelease': 's'}
    for etype in plot_df['EventType'].unique():
        subset = plot_df[plot_df['EventType'] == etype]
        ax.plot(subset['Date'], subset['RiskScore'], label=etype, color=colors[etype], marker=markers[etype], linestyle='-')
        for _, row in subset.iterrows():
            label = row['Title'] if len(row['Title']) < 30 else row['Title'][:27] + '...'
            ax.annotate(label, (row['Date'], row['RiskScore']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color=colors[etype])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Risk Score')
    ax.set_xlabel('Date')
    ax.set_title('Zilebesiran Risk Score Timeline')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

if __name__ == "__main__":
    main()
