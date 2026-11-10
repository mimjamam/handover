"""Web page for the demo. Talks to the API over HTTP only.

Run:  uv run streamlit run src/handover/app.py
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("HANDOVER_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Handover", page_icon="📨", layout="centered")


@st.cache_data(ttl=600)
def get_info() -> dict:
    return requests.get(f"{API_URL}/info", timeout=30).json()


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


st.title("Handover")
st.caption(
    "Routes customer messages to the right team, in German and English. "
    "When it is not sure, it hands the message to a person instead of guessing."
)

try:
    info = get_info()
except requests.RequestException:
    st.error("The routing service is starting. Reload the page in a few seconds.")
    st.stop()

points = [p for p in info["operating_points"] if p["threshold"] is not None]
targets = [p["target_risk"] for p in points]

st.subheader("1. How many mistakes can you accept?")
target = st.select_slider(
    "Accepted error rate among automatically routed messages",
    options=targets,
    value=info["default_target_risk"] if info["default_target_risk"] in targets else targets[0],
    format_func=pct,
)
point = next(p for p in points if p["target_risk"] == target)
c1, c2, c3 = st.columns(3)
c1.metric("Routed automatically", f"{point['test']['coverage'] * 100:.1f}%")
c2.metric("Wrong among those", f"{point['test']['risk'] * 100:.1f}%")
c3.metric("If it never handed over", f"{(1 - info['test']['accuracy']) * 100:.1f}% wrong")
st.caption(f"Measured on {info['rows']['test']:,} test messages the model never saw. The rest go to a person.")

st.subheader("2. Try a message")
examples = {e["label"]: e["text"] for e in info.get("examples", [])}
examples["DE · Unklare Nachricht"] = "Hallo, es funktioniert nicht. Können Sie mir bitte helfen?"
choice = st.selectbox(
    "Example (real messages from the test set, never seen by the model)", ["Write my own"] + list(examples)
)
text = st.text_area("Message", value=examples.get(choice, ""), height=170, max_chars=5000)

if st.button("Route message", type="primary", disabled=not text.strip()):
    try:
        response = requests.post(f"{API_URL}/route", json={"text": text, "target_risk": target}, timeout=60)
        response.raise_for_status()
    except requests.RequestException as error:
        st.error(f"The routing service did not answer: {error}")
        st.stop()
    result = response.json()

    if result["decision"] == "route":
        st.success(f"**Send to {result['team']}**  ·  {pct(result['confidence'])} confident")
    elif result["reason"] == "no similar past ticket":
        st.warning("**Hand over to a person**  ·  nothing similar enough has been seen before")
    else:
        st.warning(
            f"**Hand over to a person**  ·  best guess is only {pct(result['confidence'])} confident, "
            f"below the {pct(result['threshold'])} needed for this setting"
        )

    precedent = result["precedent"]
    if result["reason"] == "no similar past ticket":
        lead = "**Closest past message** (not similar enough to rely on) went to"
    else:
        lead = "**Why:** the most similar past message went to"
    st.markdown(f"{lead} *{precedent['team']}* (similarity {precedent['similarity']:.3f}).")
    st.info(precedent["text"])
    votes = ", ".join(f"{v['team']} ({v['neighbours']})" for v in result["votes"])
    st.caption(f"Teams of the 5 most similar past messages: {votes}. Answered in {result['latency_ms']:.0f} ms.")

with st.expander("How it works, and its limits"):
    bands = pd.DataFrame(info["similarity_bands"])
    st.markdown(
        f"""
- The system keeps {info["past_tickets_in_memory"]:,} past messages in memory. For a new message it finds the
  most similar ones and proposes the team of the closest.
- It then estimates the chance that this proposal is right, and answers only if that chance is above a
  threshold. The threshold is chosen on held-out data so the error rate among routed messages stays
  under the accepted level.
- It can only route what resembles something seen before. New kinds of message go to a person.
  That is the intended behaviour, and the table shows why.
"""
    )
    table = pd.DataFrame(
        {
            "Closest past message": bands["band"],
            "Share of messages": (bands["share_of_tickets"] * 100).round(1).astype(str) + "%",
            "Proposal correct": (bands["accuracy"] * 100).round(1).astype(str) + "%",
        }
    )
    st.table(table.set_index("Closest past message"))
    st.markdown(
        f"""
- Without handover the proposal is right {pct(info["test"]["accuracy"])} of the time
  (German {pct(info["test"]["accuracy_de"])}, English {pct(info["test"]["accuracy_en"])}).
- The data is a public, machine-generated ticket set (CC BY-NC 4.0, non-commercial) with many
  near-repeated messages. A real company's repeat rate decides how much could be automated there.
"""
    )
