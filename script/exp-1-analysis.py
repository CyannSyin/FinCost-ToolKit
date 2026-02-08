import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt

from FinCost.analysis import (
    calculate_cumulative_cost_series,
    calculate_monthly_cost_series,
    calculate_portfolio_series,
    calculate_token_cost_by_date,
)
from FinCost.commission import extract_actions_and_commission
from FinCost.config import get_project_root, load_llm_pricing, load_static_config
from FinCost.records import load_experiment_records


def _parse_date(value):
    if not value:
        return None
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    date_part = text.split(" ")[0].split("T")[0]
    try:
        return datetime.strptime(date_part, "%Y-%m-%d")
    except ValueError:
        return None


def _cumulative_sum(values):
    total = 0.0
    series = []
    for value in values:
        total += float(value)
        series.append(total)
    return series


def _load_static_configs(data_dir):
    static_map = {}
    for path in glob.glob(os.path.join(data_dir, "static-*.jsonl")):
        try:
            config = load_static_config(path)
        except ValueError:
            continue
        model = config.get("llm_model")
        if model:
            static_map[model] = config
    return static_map


def _get_model_label(records, fallback):
    if records:
        return records[0].get("model") or records[0].get("signature") or fallback
    return fallback


def compute_real_profit_series(records, model_pricing, static_config):
    initial_cash = float(static_config.get("initial_cash", 0.0))
    monthly_cost = float(static_config.get("data_subscription_monthly", 0.0))
    infra_cost_per_day = 0.2

    _, commission_by_date, _ = extract_actions_and_commission(records)
    _, token_cost_by_date = calculate_token_cost_by_date(records, model_pricing)

    dates, holding_profit_series = calculate_portfolio_series(records, initial_cash)
    portfolio_value_series = [initial_cash + value for value in holding_profit_series]

    monthly_additions, _ = calculate_monthly_cost_series(records, monthly_cost)
    cumulative_cost_series = calculate_cumulative_cost_series(
        records,
        commission_by_date,
        token_cost_by_date,
        infra_cost_per_day,
        monthly_additions,
    )

    real_profit_series = [
        portfolio_value_series[i] - cumulative_cost_series[i]
        for i in range(len(portfolio_value_series))
    ]
    return dates, real_profit_series


def main():
    project_root = get_project_root()
    data_dir = os.path.join(project_root, "data", "exp-1")
    output_dir = os.path.join(project_root, "result", "exp-1")
    os.makedirs(output_dir, exist_ok=True)

    model_pricing = load_llm_pricing(os.path.join(project_root, "config_llm.json"))
    static_configs = _load_static_configs(data_dir)

    record_paths = sorted(glob.glob(os.path.join(data_dir, "experiment_records_*.jsonl")))
    if not record_paths:
        raise FileNotFoundError(f"No experiment record files in {data_dir}")

    plt.figure(figsize=(12, 6))
    for record_path in record_paths:
        records = load_experiment_records(record_path)
        model_label = _get_model_label(records, os.path.basename(record_path))
        static_config = static_configs.get(model_label)
        if not static_config:
            print(f"Warning: missing static config for {model_label}, using defaults.")
            static_config = {"initial_cash": 10000.0, "data_subscription_monthly": 0.0}

        dates, real_profit_series = compute_real_profit_series(
            records,
            model_pricing,
            static_config,
        )
        parsed_dates = [_parse_date(value) for value in dates]
        if any(value is None for value in parsed_dates):
            parsed_dates = list(range(len(real_profit_series)))

        plt.plot(parsed_dates, real_profit_series, label=model_label)

    plt.title("Real Net Value Over Time (All Models)")
    plt.xlabel("Date")
    plt.ylabel("Net Value After Costs (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "exp-1-real-profit-timeseries.png")
    plt.savefig(output_path, dpi=200)


if __name__ == "__main__":
    main()
