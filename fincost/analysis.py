from collections import defaultdict


def calculate_token_cost_by_date(records, model_pricing):
    token_usage_by_date = defaultdict(lambda: {"input": 0, "output": 0, "cache": 0})
    token_cost_by_date = defaultdict(float)

    for record in records:
        date = record.get("date")
        llm_usage = record.get("llm_usage") or {}
        record_model = record.get("model")
        usage_model = llm_usage.get("model")
        if record_model and usage_model and record_model != usage_model:
            print(f"Warning: model mismatch in record: {record_model} vs {usage_model}")
        model = usage_model or record_model
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
    extra_additions,
):
    cumulative = 0.0
    cumulative_series = []
    for idx, record in enumerate(records):
        date = record.get("date")
        daily_cost = (
            commission_by_date.get(date, 0.0)
            + token_cost_by_date.get(date, 0.0)
            + infra_cost_per_day
            + (extra_additions[idx] if idx < len(extra_additions) else 0.0)
        )
        cumulative += daily_cost
        cumulative_series.append(cumulative)
    return cumulative_series
