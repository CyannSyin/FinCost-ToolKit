import json
import os
import random
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


# -------------------------- Commission Calculation (from demo.py) --------------------------
COMMISSION_PER_SHARE = 0.005  # Per share commission USD (IB Fixed: $0.005/share)
COMMISSION_MINIMUM = 1.0  # Minimum commission per order USD (IB Fixed: $1.00)
COMMISSION_MAXIMUM_RATE = 0.01  # Maximum commission as % of trade value (IB: 1%)


def calculate_commission(shares: int, trade_value: float) -> float:
    """
    Calculate commission based on Interactive Brokers Fixed pricing structure.
    Reference: https://www.interactivebrokers.com/cn/pricing/commissions-stocks.php

    IB Fixed Pricing:
    - Per share: $0.005 per share
    - Minimum: $1.00 per order
    - Maximum: 1% of trade value
    """
    if shares == 0 or trade_value == 0:
        return 0.0

    base_commission = shares * COMMISSION_PER_SHARE
    commission = max(base_commission, COMMISSION_MINIMUM)
    max_commission = trade_value * COMMISSION_MAXIMUM_RATE
    commission = min(commission, max_commission)
    return commission


def load_experiment_records(records_path: str):
    records = []
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_llm_pricing(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    unit = config.get("unit", "per_1k_tokens")
    if unit != "per_1k_tokens":
        raise ValueError(f"Unsupported pricing unit: {unit}")
    models = config.get("models", {})
    if not models:
        raise ValueError("No models found in config_llm.json")
    return models


def load_app_config(config_path: str):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        if not raw:
            return {}
        return json.loads(raw)


def _relaxed_json_loads(raw: str):
    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    return json.loads(cleaned)


def load_static_config(static_path: str):
    with open(static_path, "r", encoding="utf-8") as f:
        raw = f.read()
    data = _relaxed_json_loads(raw)
    structure = data.get("structure", {})
    if not structure:
        raise ValueError("Missing structure in static config")
    merged = dict(structure)
    for key, value in data.items():
        if key != "structure":
            merged[key] = value
    return merged


def calculate_token_cost_by_date(records, model_pricing):
    token_usage_by_date = defaultdict(lambda: {"input": 0, "output": 0, "cache": 0})
    token_cost_by_date = defaultdict(float)

    for record in records:
        date = record.get("date")
        llm_usage = record.get("llm_usage") or {}
        record_model = record.get("model")
        usage_model = llm_usage.get("model")
        if record_model and usage_model and record_model != usage_model:
            raise ValueError(f"Model mismatch in record: {record_model} vs {usage_model}")
        model = record_model or usage_model
        if not model:
            raise ValueError("Missing model in record llm_usage/model")
        if model not in model_pricing:
            raise ValueError(f"Model pricing not found in config_llm.json: {model}")

        input_tokens = int(llm_usage.get("input_tokens") or 0)
        output_tokens = int(llm_usage.get("output_tokens") or 0)
        cached_tokens = int(llm_usage.get("cached_tokens") or 0)

        token_usage_by_date[date]["input"] += input_tokens
        token_usage_by_date[date]["output"] += output_tokens
        token_usage_by_date[date]["cache"] += cached_tokens

        prices = model_pricing[model]
        input_cost = (input_tokens / 1000.0) * float(prices.get("input_price_per_k_tokens", 0))
        output_cost = (output_tokens / 1000.0) * float(prices.get("output_price_per_k_tokens", 0))
        cache_cost = (cached_tokens / 1000.0) * float(prices.get("cache_price_per_k_tokens", 0))
        token_cost_by_date[date] += input_cost + output_cost + cache_cost

    return token_usage_by_date, token_cost_by_date


def extract_actions_and_commission(records):
    actions_by_date = {}
    commission_by_date = defaultdict(float)
    commission_total = 0.0

    for record in records:
        date = record.get("date")
        trades = record.get("trades", [])
        actions = []

        for trade in trades:
            decision_type = trade.get("decision_type", "UNKNOWN")
            actions.append(decision_type)

            quantity = int(trade.get("quantity") or 0)
            price = trade.get("execution_price")
            if price is None:
                price = trade.get("analysis_price")
            if price is None:
                trade_value = 0.0
            else:
                trade_value = float(price) * quantity

            # Commission is calculated per trade (per order), not aggregated daily.
            trade_commission = calculate_commission(quantity, trade_value)
            commission_by_date[date] += trade_commission
            commission_total += trade_commission

        actions_by_date[date] = actions

    return actions_by_date, commission_by_date, commission_total


def print_action_summary(actions_by_date):
    print("Action summary by trading day:")
    for date in sorted(actions_by_date.keys()):
        counts = Counter(actions_by_date[date])
        counts_str = ", ".join([f"{k}={v}" for k, v in counts.items()])
        print(f"  {date}: {counts_str}")


def plot_cost_pie(
    commission_total,
    token_total,
    infra_total,
    monthly_total,
    uncertain_total,
    output_path,
    include_monthly=True,
):
    labels = ["commission", "token", "infra", "monthly", "uncertain"]
    values = [commission_total, token_total, infra_total, monthly_total, uncertain_total]
    colors = ["#809bce", "#eac4d5", "#b8e0d4", "#95b8d1", "#f5e2ea"]
    if not include_monthly:
        filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if l != "monthly"]
        labels, values, colors = zip(*filtered) if filtered else ([], [], [])
    total = sum(values)
    if total <= 0:
        print("No costs to plot.")
        return
    formatted_labels = [f"{label} (${value:.2f})" for label, value in zip(labels, values)]

    def make_autopct(vals):
        def _autopct(pct):
            value = pct * sum(vals) / 100.0
            return f"{pct:.1f}%\n${value:.2f}"
        return _autopct

    plt.figure(figsize=(6, 6))
    plt.pie(
        values,
        labels=formatted_labels,
        autopct=make_autopct(values),
        startangle=45,
        colors=colors,
    )
    plt.title("Cost share: commission vs token vs infra")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved pie chart to: {output_path}")


def calculate_portfolio_series(records, initial_cash):
    cash = float(initial_cash)
    holdings = defaultdict(int)
    last_price = {}
    dates = []
    holding_profit_series = []

    for record in records:
        date = record.get("date")
        trades = record.get("trades", [])
        for trade in trades:
            decision_type = trade.get("decision_type", "").upper()
            ticker = trade.get("ticker") or ""
            quantity = int(trade.get("quantity") or 0)
            price = trade.get("execution_price")
            if price is None:
                price = trade.get("analysis_price")
            if price is None:
                price = 0.0
            price = float(price)

            if ticker:
                last_price[ticker] = price

            if decision_type == "BUY":
                cash -= price * quantity
                holdings[ticker] += quantity
            elif decision_type == "SELL":
                cash += price * quantity
                holdings[ticker] -= quantity

        holdings_value = sum(qty * last_price.get(ticker, 0.0) for ticker, qty in holdings.items())
        portfolio_value = cash + holdings_value
        holding_profit = portfolio_value - float(initial_cash)

        dates.append(date)
        holding_profit_series.append(holding_profit)

    return dates, holding_profit_series


def calculate_monthly_cost_series(records, monthly_cost):
    monthly_additions = []
    last_month = None
    total = 0.0
    for record in records:
        date_str = str(record.get("date") or "")
        date_part = date_str.split(" ")[0].split("T")[0]
        month_key = date_part[:7] if len(date_part) >= 7 else None
        add_cost = 0.0
        if month_key and (last_month is None or month_key != last_month):
            add_cost = monthly_cost
            total += monthly_cost
        monthly_additions.append(add_cost)
        if month_key:
            last_month = month_key
    return monthly_additions, total


def calculate_cumulative_cost_series(
    records,
    commission_by_date,
    token_cost_by_date,
    infra_cost_per_day,
    monthly_additions,
):
    cumulative = 0.0
    cumulative_series = []
    for idx, record in enumerate(records):
        date = record.get("date")
        daily_cost = (
            commission_by_date.get(date, 0.0)
            + token_cost_by_date.get(date, 0.0)
            + infra_cost_per_day
            + (monthly_additions[idx] if idx < len(monthly_additions) else 0.0)
        )
        cumulative += daily_cost
        cumulative_series.append(cumulative)
    return cumulative_series


def plot_performance_lines(dates, holding_profit_series, cumulative_cost_series, real_profit_series, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(
        dates,
        holding_profit_series,
        label="Holding Profit (holdings + cash - initial cash)",
        color="#88a4c9",
    )
    plt.plot(
        dates,
        cumulative_cost_series,
        label="Cumulative Cost (token + infra + commission)",
        color="#ff8696",
    )
    plt.plot(
        dates,
        real_profit_series,
        label="Real Profit (holding profit - cost)",
        color="#ff8906",
    )
    plt.axhline(0, color="black", linewidth=2.5, linestyle="--")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved performance chart to: {output_path}")


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    app_config_path = os.path.join(os.path.dirname(__file__), "config.json")
    app_config = load_app_config(app_config_path)
    static_path = app_config.get(
        "static_path",
        os.path.join(data_dir, "static_gpt-4o-mini_10000.jsonl"),
    )
    static_config = load_static_config(static_path)
    records_path = app_config.get(
        "records_path",
        os.path.join(
            data_dir,
            "experiment_records_gpt-4o-mini_10000.0_2026-01-06-14-00-00_2026-01-07-14-00-00.jsonl",
        ),
    )
    records = load_experiment_records(records_path)
    pricing_path = os.path.join(os.path.dirname(__file__), "config_llm.json")
    model_pricing = load_llm_pricing(pricing_path)

    actions_by_date, commission_by_date, commission_total = extract_actions_and_commission(records)
    print_action_summary(actions_by_date)

    trading_days = len(actions_by_date)
    _, token_cost_by_date = calculate_token_cost_by_date(records, model_pricing)
    token_total = sum(token_cost_by_date.values())
    infra_cost_per_day = 0.2

    monthly_cost = float(static_config.get("data_subscription_monthly", 0.0))
    uncertain_cost = random.uniform(0.0, 0.5)
    monthly_additions, monthly_total = calculate_monthly_cost_series(records, monthly_cost)
    infra_total = trading_days * infra_cost_per_day
    total_cost = commission_total + token_total + infra_total + monthly_total + uncertain_cost

    report_lines = [
        "Action summary by trading day:",
        *[
            f"  {date}: {', '.join([f'{k}={v}' for k, v in Counter(actions_by_date[date]).items()])}"
            for date in sorted(actions_by_date.keys())
        ],
        "",
        "Cost totals:",
        f"  trading_days: {trading_days}",
        f"  commission_total: {commission_total:.2f}",
        f"  token_total: {token_total:.2f}",
        f"  infra_total: {infra_total:.2f}",
        f"  monthly_total: {monthly_total:.2f}",
        f"  uncertain_cost: {uncertain_cost:.2f}",
        f"  total_cost: {total_cost:.2f}",
    ]
    report_text = "\n".join(report_lines)
    print(report_text)

    llm_model = str(static_config.get("llm_model", "unknown"))
    initial_cash = str(static_config.get("initial_cash", "unknown"))
    frequency = str(
        static_config.get("decision_frequency", static_config.get("frequency", "unknown"))
    )
    result_dir = os.path.join(
        os.path.dirname(__file__),
        "result",
        f"{llm_model}-{initial_cash}-{frequency}",
    )
    os.makedirs(result_dir, exist_ok=True)
    txt_filename = f"{llm_model}-{initial_cash}-{frequency}.txt"
    txt_path = os.path.join(result_dir, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nSaved report to: {txt_path}")

    jsonl_filename = f"{llm_model}_{initial_cash}_{frequency}.jsonl"
    result_path = os.path.join(result_dir, jsonl_filename)
    report_payload = {
        "llm_model": llm_model,
        "initial_cash": initial_cash,
        "frequency": frequency,
        "action_summary_by_day": [
            {
                "date": date,
                "actions": dict(Counter(actions_by_date[date])),
            }
            for date in sorted(actions_by_date.keys())
        ],
        "cost_totals": {
            "trading_days": trading_days,
            "commission_total": round(commission_total, 2),
            "token_total": round(token_total, 2),
            "infra_total": round(infra_total, 2),
            "monthly_total": round(monthly_total, 2),
            "uncertain_cost": round(uncertain_cost, 2),
            "total_cost": round(total_cost, 2),
        },
    }
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(report_payload, ensure_ascii=False) + "\n")
    print(f"\nSaved report to: {result_path}")

    pie_dir = os.path.join(result_dir, "pie-chart")
    os.makedirs(pie_dir, exist_ok=True)
    pie_filename = f"{llm_model}_{llm_model}_{initial_cash}_{frequency}_pie_chart.pdf"
    output_path = os.path.join(pie_dir, pie_filename)
    plot_cost_pie(
        commission_total,
        token_total,
        infra_total,
        monthly_total,
        uncertain_cost,
        output_path,
    )
    pie_filename_no_monthly = (
        f"{llm_model}_{llm_model}_{initial_cash}_{frequency}_pie_chart_no_monthly.pdf"
    )
    output_path_no_monthly = os.path.join(pie_dir, pie_filename_no_monthly)
    plot_cost_pie(
        commission_total,
        token_total,
        infra_total,
        0.0,
        uncertain_cost,
        output_path_no_monthly,
        include_monthly=False,
    )

    dates, holding_profit_series = calculate_portfolio_series(records, initial_cash)
    uncertain_additions = [uncertain_cost] + [0.0 for _ in records[1:]]
    combined_additions = [
        monthly_additions[i] + uncertain_additions[i] for i in range(len(monthly_additions))
    ]
    cumulative_cost_series = calculate_cumulative_cost_series(
        records,
        commission_by_date,
        token_cost_by_date,
        infra_cost_per_day,
        combined_additions,
    )
    real_profit_series = [
        holding_profit_series[i] - cumulative_cost_series[i]
        for i in range(len(holding_profit_series))
    ]
    line_dir = os.path.join(result_dir, "line chart")
    os.makedirs(line_dir, exist_ok=True)
    line_filename = f"{llm_model}_{llm_model}_{initial_cash}_{frequency}_line_chart.pdf"
    line_output_path = os.path.join(line_dir, line_filename)
    plot_performance_lines(
        dates,
        holding_profit_series,
        cumulative_cost_series,
        real_profit_series,
        line_output_path,
    )

    no_monthly_additions = [0.0 for _ in records]
    combined_additions_no_monthly = [
        no_monthly_additions[i] + uncertain_additions[i]
        for i in range(len(no_monthly_additions))
    ]
    cumulative_cost_series_no_monthly = calculate_cumulative_cost_series(
        records,
        commission_by_date,
        token_cost_by_date,
        infra_cost_per_day,
        combined_additions_no_monthly,
    )
    real_profit_series_no_monthly = [
        holding_profit_series[i] - cumulative_cost_series_no_monthly[i]
        for i in range(len(holding_profit_series))
    ]
    line_filename_no_monthly = (
        f"{llm_model}_{llm_model}_{initial_cash}_{frequency}_line_chart_no_monthly.pdf"
    )
    line_output_path_no_monthly = os.path.join(line_dir, line_filename_no_monthly)
    plot_performance_lines(
        dates,
        holding_profit_series,
        cumulative_cost_series_no_monthly,
        real_profit_series_no_monthly,
        line_output_path_no_monthly,
    )


if __name__ == "__main__":
    main()
