import json
from collections import defaultdict

from FinCost.commission import calculate_commission

INFRA_COST_PER_DAY = 0.2


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_llm_pricing(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    unit = config.get("unit", "per_1k_tokens")
    if unit != "per_1k_tokens":
        raise ValueError(f"Unsupported pricing unit: {unit}")
    models = config.get("models", {})
    if not models:
        raise ValueError("No models found in config_llm.json")
    return models


def extract_price_dates(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            series_key = next((k for k in data if k.startswith("Time Series")), None)
            series = data.get(series_key, {}) if series_key else {}
            dates = set()
            for ts in series.keys():
                dates.add(str(ts).split(" ")[0].split("T")[0])
            return sorted(dates)
    return []


def record_date_key(record):
    value = record.get("date")
    if not value:
        return None
    text = str(value)
    return text.split(" ")[0].split("T")[0]


def record_model_name(record):
    llm_usage = record.get("llm_usage") or {}
    record_model = record.get("model")
    usage_model = llm_usage.get("model")
    if record_model and usage_model and record_model != usage_model:
        print(f"Warning: model mismatch in record: {record_model} vs {usage_model}")
    return usage_model or record_model


def token_cost_for_record(record, model_pricing):
    llm_usage = record.get("llm_usage") or {}
    model = record_model_name(record)
    if not model:
        return 0.0
    if model not in model_pricing:
        raise ValueError(f"Model pricing not found in config_llm.json: {model}")
    input_tokens = int(llm_usage.get("input_tokens") or 0)
    output_tokens = int(llm_usage.get("output_tokens") or 0)
    cached_tokens = int(llm_usage.get("cached_tokens") or 0)
    prices = model_pricing[model]
    input_cost = (input_tokens / 1000.0) * float(prices.get("input_price_per_k_tokens", 0))
    output_cost = (output_tokens / 1000.0) * float(prices.get("output_price_per_k_tokens", 0))
    cache_cost = (cached_tokens / 1000.0) * float(prices.get("cache_price_per_k_tokens", 0))
    return input_cost + output_cost + cache_cost


def commission_for_record(record):
    total = 0.0
    for trade in record.get("trades", []):
        decision_type = str(trade.get("decision_type") or "").upper()
        if decision_type not in {"BUY", "SELL"}:
            continue
        quantity = int(trade.get("quantity") or 0)
        price = trade.get("execution_price")
        if price is None:
            price = trade.get("analysis_price")
        trade_value = None if price is None else float(price) * quantity
        total += calculate_commission(quantity, trade_value)
    return total


def aggregate_daily_cost(records, model_pricing, date_filter=None, date_cutoff=None):
    daily = defaultdict(lambda: {"token": 0.0, "commission": 0.0, "record_count": 0})
    for record in records:
        date_key = record_date_key(record)
        if not date_key:
            continue
        if date_cutoff and date_key > date_cutoff:
            continue
        if date_filter and date_key not in date_filter:
            continue
        daily[date_key]["token"] += token_cost_for_record(record, model_pricing)
        daily[date_key]["commission"] += commission_for_record(record)
        daily[date_key]["record_count"] += 1
    return daily


def write_daily_costs(output_path, entries):
    with open(output_path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def build_entries(daily_costs, frequency, model_name, date_order):
    entries = []
    cumulative = 0.0
    for date_key in date_order:
        if date_key not in daily_costs:
            continue
        token_cost = daily_costs[date_key]["token"]
        commission_cost = daily_costs[date_key]["commission"]
        infra_cost = INFRA_COST_PER_DAY
        total_cost = token_cost + commission_cost + infra_cost
        cumulative += total_cost
        entries.append(
            {
                "date": date_key,
                "frequency": frequency,
                "model": model_name,
                "token_cost": round(token_cost, 6),
                "commission_cost": round(commission_cost, 6),
                "infra_cost": round(infra_cost, 6),
                "total_cost": round(total_cost, 6),
                "cumulative_cost": round(cumulative, 6),
                "record_count": daily_costs[date_key]["record_count"],
            }
        )
    return entries


def main():
    pricing = load_llm_pricing("config_llm.json")

    hourly_price_dates = extract_price_dates("data/merged-hourly.jsonl")
    daily_price_dates = extract_price_dates("data/merged-daily.jsonl")

    gpt_hourly_records = load_jsonl(
        "data/exp-3/experiment_records_gpt-5.2_10000.0_2025-12-01-10-00-00_2025-12-31-15-00-00.jsonl"
    )
    deepseek_hourly_records = load_jsonl(
        "data/exp-3/experiment_records_DeepSeek-V3_10000.0_2025-12-01-10-00-00_2025-12-31-15-00-00.jsonl"
    )

    gpt_daily_records = load_jsonl(
        "data/exp-1/experiment_records_gpt-5.2_10000.0_2025-12-01_2026-01-30.jsonl"
    )
    deepseek_daily_records = load_jsonl(
        "data/exp-1/experiment_records_DeepSeek-V3_10000.0_2025-12-01_2026-01-30.jsonl"
    )

    hourly_gpt_costs = aggregate_daily_cost(
        gpt_hourly_records, pricing, date_filter=set(hourly_price_dates)
    )
    hourly_deepseek_costs = aggregate_daily_cost(
        deepseek_hourly_records, pricing, date_filter=set(hourly_price_dates)
    )

    daily_cutoff = "2025-12-31"
    daily_gpt_costs = aggregate_daily_cost(
        gpt_daily_records,
        pricing,
        date_filter=set(daily_price_dates),
        date_cutoff=daily_cutoff,
    )
    daily_deepseek_costs = aggregate_daily_cost(
        deepseek_daily_records,
        pricing,
        date_filter=set(daily_price_dates),
        date_cutoff=daily_cutoff,
    )

    entries = []
    entries.extend(build_entries(hourly_gpt_costs, "hourly", "gpt-5.2", hourly_price_dates))
    entries.extend(
        build_entries(hourly_deepseek_costs, "hourly", "DeepSeek-V3", hourly_price_dates)
    )
    entries.extend(
        build_entries(daily_gpt_costs, "daily", "gpt-5.2", daily_price_dates)
    )
    entries.extend(
        build_entries(daily_deepseek_costs, "daily", "DeepSeek-V3", daily_price_dates)
    )

    output_path = "daily-hourly-compare-output.jsonl"
    write_daily_costs(output_path, entries)
    print(f"Wrote daily cost output to {output_path}")


if __name__ == "__main__":
    main()
