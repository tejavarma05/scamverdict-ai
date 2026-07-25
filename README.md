# 🛡️ ScamVerdict AI

Detects fraudulent job postings from their text, and shows which words drove the
decision.

**Live demo:** _add your Streamlit URL here after deploying_

---

## Why job postings

Job scams are a live problem — fake listings harvest personal data, charge bogus
"equipment deposits," or recruit money mules. Job seekers are the least equipped
people to spot them, because a scam posting is designed to look exactly like the
listings they're already reading.

A model is only useful here if the person can see *why* something was flagged.
"Trust me, 87%" doesn't help anyone decide whether to send their passport scan.

## Why accuracy is not reported

Roughly **5% of postings in this dataset are fraudulent**. A model that predicts
"legitimate" for every posting scores **95% accuracy** while catching zero scams.

Accuracy is therefore worse than useless here — it rewards doing nothing. This
project reports **precision, recall, and PR-AUC**, and the training script prints the
trivial baseline alongside the real numbers so they can't be read out of context.

See `models/metrics.json` for the current held-out test results.

## Approach

**TF-IDF (word 1–2 grams) + logistic regression with balanced class weights.**

That is a deliberate choice, not a shortcut:

- It's a genuinely strong baseline for text classification — transformers buy less
  here than people assume, and cost far more to serve.
- It trains in seconds, so iterating on features is cheap.
- **It's exactly explainable.** For a linear model on TF-IDF, each token's push toward
  "scam" is `coefficient × tf-idf value`. The per-token breakdown in the app is the
  literal arithmetic of the decision, not a post-hoc approximation like SHAP or LIME.

Two details that matter more than the model choice:

- **Stratified splits** — three-way train/validation/test, each preserving the ~5%
  positive rate. A random split can hand you a test set with almost no positives.
- **Tuned decision threshold** — chosen by maximizing F1 on the validation split
  rather than left at 0.5. On imbalanced problems 0.5 is arbitrary and usually wrong.
  The tuned threshold ships with the model.

## Running it

```bash
git clone https://github.com/<you>/scamverdict-ai && cd scamverdict-ai
pip install -r requirements.txt
```

Download the dataset (free Kaggle account required) and place the CSV at
`data/fake_job_postings.csv`:
<https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction>

```bash
python src/train.py --data data/fake_job_postings.csv
streamlit run app.py
```

## Layout

```
src/train.py     Training, evaluation, threshold tuning, metrics export
app.py           Streamlit UI with per-token explanations
models/          Trained pipeline + metrics.json (committed so the demo runs)
```

## Limitations

- Trained on one public dataset from a single time period. Scam language moves fast;
  a model trained on 2020 postings will degrade against 2026 tactics.
- Bag-of-words has no word order. "We will never ask for a wire transfer" and
  "send a wire transfer" share their strongest tokens.
- Text only. Domain age, employer verification, and posting metadata are stronger
  signals in practice and aren't used here.
- **This is a screening aid, not a verdict.** A high score means look closer. It is
  not evidence that a specific employer is fraudulent.

## Possible next steps

- Calibrate probabilities (Platt / isotonic) so the score reads as a real likelihood
- Compare against a fine-tuned transformer to quantify what the simple model gives up
- Add URL and domain-age features for the phishing-link case
