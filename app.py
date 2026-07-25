"""ScamVerdict AI - paste a job posting, get a risk score and the reasons behind it.

The explanation is not decorative. For a linear model on TF-IDF features, each
token's push toward "scam" is exactly coefficient x tf-idf value, so the per-token
breakdown below is the literal arithmetic of the decision -- not a post-hoc guess.
That is the whole argument for using a linear model here.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

MODEL_PATH = Path("models/scamverdict.joblib")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("scamverdict.joblib")
METRICS_PATH = Path("models/metrics.json")
if not METRICS_PATH.exists():
    METRICS_PATH = Path("metrics.json")


st.set_page_config(page_title="ScamVerdict AI", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


def explain(pipeline, text: str, top_n: int = 10):
    """Return the tokens that pushed the score most, in both directions."""
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    x = vec.transform([text])
    names = vec.get_feature_names_out()
    coefs = clf.coef_[0]

    idx = x.nonzero()[1]
    if len(idx) == 0:
        return [], []

    contributions = [(names[i], float(x[0, i] * coefs[i])) for i in idx]
    contributions.sort(key=lambda p: p[1], reverse=True)

    toward_scam = [c for c in contributions if c[1] > 0][:top_n]
    toward_legit = [c for c in contributions if c[1] < 0][-top_n:][::-1]
    return toward_scam, toward_legit


st.title("🛡️ ScamVerdict AI")
st.caption("Fake job posting detection with explainable risk scoring")

bundle = load_model()
if bundle is None:
    st.error(
        "No trained model found. Run `python src/train.py --data data/fake_job_postings.csv` "
        "first, then restart the app."
    )
    st.stop()

pipeline = bundle["pipeline"]
threshold = bundle["threshold"]
metrics = load_metrics()

EXAMPLE = (
    "URGENT HIRING - Data Entry Clerk - Work From Home\n\n"
    "No experience necessary! Earn $4,500 weekly working just 2 hours a day. "
    "We are hiring immediately, no interview required. To begin processing your "
    "onboarding, please send a $150 equipment deposit via wire transfer. "
    "Contact us on Telegram for faster response."
)

text = st.text_area(
    "Paste a job posting",
    height=220,
    placeholder="Paste the full posting text here...",
)

col_a, col_b = st.columns([1, 1])
with col_a:
    check = st.button("Analyze posting", type="primary", use_container_width=True)
with col_b:
    if st.button("Load example scam", use_container_width=True):
        st.session_state["example"] = EXAMPLE
        st.rerun()

if "example" in st.session_state and not text:
    text = st.session_state["example"]
    st.info("Example loaded below — press Analyze.")
    st.code(text[:400] + "...", language=None)

if check and text.strip():
    if len(text.split()) < 15:
        st.warning("That's very short. Scores on a few words aren't meaningful — paste the full posting.")

    score = float(pipeline.predict_proba([text])[0, 1])
    flagged = score >= threshold

    st.divider()
    if flagged:
        st.error(f"### ⚠️ Likely scam — risk score {score:.1%}")
    else:
        st.success(f"### ✅ Looks legitimate — risk score {score:.1%}")
    st.progress(min(score, 1.0))
    st.caption(f"Flagged when score ≥ {threshold:.1%} (threshold tuned on a validation split, not left at 50%)")

    toward_scam, toward_legit = explain(pipeline, text)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Pushed toward scam**")
        if toward_scam:
            for token, weight in toward_scam:
                st.markdown(f"- `{token}` &nbsp; +{weight:.3f}", unsafe_allow_html=True)
        else:
            st.caption("No strong scam signals found.")
    with c2:
        st.markdown("**Pushed toward legitimate**")
        if toward_legit:
            for token, weight in toward_legit:
                st.markdown(f"- `{token}` &nbsp; {weight:.3f}", unsafe_allow_html=True)
        else:
            st.caption("No strong legitimacy signals found.")

    st.caption(
        "Weights are coefficient × TF-IDF value — the actual arithmetic behind the score."
    )

if metrics:
    st.divider()
    with st.expander("Model performance (held-out test set)"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{metrics['precision_fraudulent']:.1%}")
        m2.metric("Recall", f"{metrics['recall_fraudulent']:.1%}")
        m3.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
        cm = metrics["confusion_matrix"]
        st.markdown(
            f"Tested on **{metrics['n_test']:,}** held-out postings, of which "
            f"**{metrics['positive_rate']:.1%}** are fraudulent.\n\n"
            f"Caught **{cm['tp']}** scams, missed **{cm['fn']}**, "
            f"raised **{cm['fp']}** false alarms."
        )
        st.warning(
            f"Accuracy is deliberately not shown. With only "
            f"{metrics['positive_rate']:.1%} of postings fraudulent, always predicting "
            f"\"legitimate\" scores {metrics['trivial_baseline_accuracy']:.1%} accuracy while "
            "catching zero scams. Precision and recall are the metrics that mean something here."
        )

st.divider()
st.caption(
    "Screening aid, not a verdict. Treat a high score as a reason to check further — "
    "never as proof an employer is fraudulent."
)
