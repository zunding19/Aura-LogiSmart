# Aura: early warning for pharmaceutical shipments

**Dormathon 2026, Track 1: Predict Early, Decide Better** (organised by AI Hackerdorm)

**Team Aura:** Han Zun Ding, Chen Zhe Tein, Ryan Hew Ka Shing

---

## The problem

A healthcare logistics coordinator watches dozens of pharmaceutical shipments at once. Today they learn that a batch was delayed or spoiled only after it arrives. By then the medicine is lost and the hospital is short.

## Who it is for

One user: the logistics coordinator who decides which shipment to act on, and what to do about it, while the shipment is still moving.

## What Aura does

Aura scores every active shipment and walks the coordinator through five steps.

| Step | Question | What Aura shows |
|---|---|---|
| Detect | Will it be disrupted? | A disruption risk from 0 to 100%, in four tiers (Low, Medium, High, Critical) |
| Categorize | What kind of trouble? | Transport delay, cold-chain excursion, or a delay leading to an excursion |
| Prioritize | Which shipment first? | A queue ranked by risk, cargo value, hospital urgency and thermal buffer left |
| Explain | Why? | The top 3 reasons in plain words, for example "Hurricane on the route. In clear weather the risk would be 46% instead of 99%." |
| Decide | What should we do? | Each action with its fee, the risk before and after, the net saving and the thermal runway gained |

It also includes a simulated Smart SLA (a breach notice and escrow lock when risk is above 80% and the buffer is under 3 hours) and a digital custody handshake that releases the escrow once a corrective action is logged.

## Does the model work?

Measured on 1,000 shipments the model never saw during training.

| Measure | Result |
|---|---|
| Precision (alerts that are real) | 85% |
| Recall (disruptions caught) | 70% |
| F1 score | 0.77 |
| ROC AUC | 0.83 |
| False alarms per 1,000 shipments | 77, against 387 if every shipment is flagged |

The model also holds up when its inputs are imperfect: scores barely move with 30% noise on the numbers, a backup model takes over when the weather is missing, and bad input returns "Insufficient data for reliable prediction" and never crashes. The full evidence is in `notebooks/05_final_model_and_evaluation.ipynb`.

## What is real and what is assumed

- **The shipments are simulated.** The dataset is general cargo (5,000 shipments, 1,006 of them pharmaceuticals) and looks generated. The scores show that the method works, not how it would score on a real carrier's records.
- **Aura covers air and road shipments.** Sea and rail are out of scope. The model still learns from all four modes, because mode does not change how often a shipment is disrupted. On air and road test shipments alone it scores precision 86% and recall 74%.
- **The model predicts disruption, not spoilage.** The data records only whether a shipment was disrupted. Thermal runway (buffer hours left against hours still to go) shows whether the cargo would survive that disruption.
- **Cargo value, storage standard, thermal buffer and hospital urgency are simulated** by `src/build_dataset.py`. They are used for ranking and money, never as model inputs.
- **Action fees, action effects and the cost of a disruption (10% of cargo value) are assumptions.** A real team would replace them with its own figures.
- **Route risk** uses a placeholder table. A live source such as ACLED can be plugged in.

## Run it

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Run the tests with `python -m pytest`.

## Use the backend directly

```python
from src.intake import assess_order

result = assess_order({
    "Place_of_Origin": "Frankfurt CargoCity",
    "Place_of_Destination": "Kuala Lumpur General Hospital",
    "Transport_Mode": "Air",
    "Lead-Time in days": 3,
    "Transit_Duration": 60,          # hours elapsed
    "Buffer_of_Package": 2.8,        # thermal hold hours left
    "Cargo_Value_USD": 85000,
    "Hospital_Urgency_Tier": "Critical Emergency",
    "Weather": "Storm",              # optional
})
```

`result` holds `risk`, `failure_mode`, `priority`, `explanation`, `thermal`, `actions`, `sla` and `scorecard`.

## Repository

```text
app.py                 Streamlit app
src/
  intake.py            One order in, one full assessment out
  model.py             Loads the saved model and scores shipments
  features.py          Builds the model's features
  explain.py           Plain-language reasons for a risk score
  prioritize.py        Triage queue
  recommendations.py   Actions, fees, net savings and the Smart SLA
  protocols.py         Custody handshake and audit trail
  build_dataset.py     Rebuilds the training file from the raw file
notebooks/             01 to 05: data checks, features, model choice, final evaluation
models/                aura_final_model.pkl, written by notebook 05
data/                  Raw file and the rebuilt training file
tests/                 63 tests covering the model, the order flow and the SLA
```

## Technology

Python 3.13, Streamlit, pandas, NumPy, scikit-learn (logistic regression) and Plotly.
