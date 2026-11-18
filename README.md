# Nassau Candy · Factory Reallocation & Shipping Optimizer

Streamlit decision-intelligence dashboard (glassmorphism UI) that predicts shipping lead time,
simulates moving each product to alternative factories, and ranks reassignment recommendations.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub. Keep `data/Nassau Candy Distributor.csv` in the repository;
   the app loads it using a path relative to `config.py`.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and select **Create app**.
3. Choose this repository, the `main` branch, and `app.py` as the app file, then deploy.

The app's dependencies are listed in `requirements.txt`, and its theme is configured in
`.streamlit/config.toml`. Do not commit credentials or `.streamlit/secrets.toml`; add deployment
secrets in the app's Streamlit Cloud settings if needed.

## Project layout

| File | Purpose |
|------|---------|
| `config.py` | Factory coordinates, product → factory mapping, state/province centroids |
| `data_prep.py` | Date parsing, lead time, distance, outlier removal, feature matrix |
| `model.py` | Trains Ridge / Random Forest / Gradient Boosting, reports RMSE, MAE, R², picks best |
| `optimizer.py` | Scenario simulation, scoring, risk flags, KPIs, route clustering, congestion |
| `style.py` | Glassmorphism CSS, HTML card helpers, Plotly theme |
| `app.py` | Streamlit app: simulator, what-if, recommendations, risk panel, clusters, model |
| `data/Nassau Candy Distributor.csv` | Source data |

## Assumptions

- The CSV has no factory column; the current product → factory mapping in `config.py` follows the
  public Nassau Candy product table. Edit it if your brief differs.
- Customer location is the state/province centroid (the data has no coordinates).
- Order/Ship dates mix `dd-mm-yyyy` and `d/m/yyyy`; both are parsed day-first.
- Profit impact uses an adjustable freight-cost assumption ($ per unit per 1,000 miles) from the sidebar.
- Lead times in this dataset are ~900–1,640 days and depend mainly on order timing, so predicted
  gains from switching factories are small. The app reports them as they are.

## Deliverables (from the Problem Overview)

| Deliverable | Location |
|-------------|----------|
| Streamlit dashboard (live analytics) | `streamlit run app.py` |
| Research paper (EDA, insights, recommendations) | `reports/research_paper.pdf` (+ `.html`) |
| Executive summary for government stakeholders | `reports/executive_summary.pdf` (+ `.html`) |

Regenerate both reports from the current data with `python reports/build_reports.py`
(PDF export uses a locally installed Microsoft Edge or Google Chrome). The reports can also be
downloaded from the dashboard's **Model & Data** tab.
