# FinCost-Demo

Cost and performance reporting for LLM-driven trading backtests.

## What It Does

`FinCost` loads experiment records and static config, calculates trading costs
(commission, token usage, infra, monthly subscription, and an uncertain add-on),
then outputs summary reports and charts.

## Quick Start (uv)

1. Create and activate a virtual environment:
   - `uv venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `uv pip install -e .`
3. Configure inputs:
   - `config.json` points to the experiment records and static config
   - `config_llm.json` contains per-model token pricing
4. Run:
   - `python main.py`

## Inputs

- `config.json`
  - `records_path`: JSONL experiment records
  - `static_path`: JSON-like static config (relaxed JSON allowed)
- `config_llm.json`
  - `unit` must be `per_1k_tokens`
  - `models` pricing with input/output/cache rates
- `data/*.jsonl`
  - Records include `date`, `model` or `llm_usage.model`, `llm_usage` token fields,
    and `trades` with `decision_type`, `quantity`, and prices.
  - Static config includes `structure` plus fields like `llm_model`,
    `initial_cash`, `decision_frequency`, and `data_subscription_monthly`.

## Outputs

Results are written under `result/<llm_model>-<initial_cash>-<frequency>/`:

- `*.txt` report with action summary and cost totals
- `*.jsonl` report payload
- `pie-chart/*.pdf` cost breakdown (with/without monthly)
- `line chart/*.pdf` performance vs costs (with/without monthly)

## Notes

- `main.py` at repo root delegates to `FinCost.main`.
- Infra cost is fixed at `0.2` per trading day.
- Uncertain cost is randomly sampled in `FinCost.main`.

