"""
exp3 vs exp1 对比分析：按日统计
时间对齐：2025-12-01 ~ 2025-12-31
输出一张表：模型, 频次, 日期, gross_profit, cost, net_profit（每日时间维度）
"""
import json
import os
import random
from collections import defaultdict

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
EXP1_DIR = os.path.join(DATA_DIR, "KDD", "exp-1")
EXP3_DIR = os.path.join(DATA_DIR, "KDD", "exp-3")
CONFIG_LLM = os.path.join(ROOT, "config_llm.json")
MERGED_DAILY = os.path.join(DATA_DIR, "merged-daily.jsonl")

# 时间范围：12/1 - 12/31
DATE_START = "2025-12-01"
DATE_END = "2025-12-31"
INITIAL_CASH = 100000.0

# 佣金参数（与 fincost/commission 一致）
COMMISSION_PER_SHARE = 0.005
COMMISSION_MINIMUM = 1.0
COMMISSION_MAXIMUM_RATE = 0.01

# 与 fincost/main 一致：infra 按每条记录 0.2；monthly 来自 static 的 data_subscription_monthly；uncertain 可选（默认 0 可复现）
INFRA_COST_PER_RECORD = 0.2


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_daily_buy_prices(prices_path):
    """与 fincost/records.load_daily_buy_prices 一致：prices_by_date[date][symbol] = price"""
    prices_by_date = defaultdict(dict)
    with open(prices_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            meta = payload.get("Meta Data", {})
            symbol = meta.get("2. Symbol") or meta.get("symbol")
            if not symbol:
                continue
            series = payload.get("Time Series (Daily)", {})
            for date, day_data in series.items():
                if not isinstance(day_data, dict):
                    continue
                price_text = day_data.get("1. buy price") or day_data.get("buy_price")
                if price_text is None:
                    continue
                try:
                    price_value = float(price_text)
                except (TypeError, ValueError):
                    continue
                prices_by_date[str(date)][str(symbol)] = price_value
    return prices_by_date


def in_range_daily(date_str):
    return DATE_START <= date_str <= DATE_END


def in_range_hourly(date_str):
    d = date_str[:10] if date_str else ""
    return DATE_START <= d <= DATE_END


def filter_records(records, is_hourly=False):
    pred = in_range_hourly if is_hourly else in_range_daily
    return [r for r in records if pred(r.get("date", ""))]


def get_day(record_date):
    """记录日期 -> 日维度日期。日频为 '2025-12-02'，小时频为 '2025-12-01 11:00:00' -> '2025-12-01'"""
    s = str(record_date or "")
    return s[:10] if " " in s or "T" in s else s


def compute_uncertain_total(records, seed=None):
    """与 fincost 一致：按 trading_days 个数，每个 random.uniform(0, 0.5) 求和。seed 用于可复现。"""
    trading_days = len(set(get_day(r.get("date")) for r in records if r.get("date")))
    if trading_days == 0:
        return 0.0
    if seed is not None:
        random.seed(seed)
    return sum(random.uniform(0.0, 0.5) for _ in range(trading_days))


def load_llm_pricing(path=CONFIG_LLM):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("models", {})


def load_static_monthly(static_path):
    """从 static 文件读 data_subscription_monthly（与 fincost 一致）；支持含注释的 JSON/JSONL"""
    try:
        with open(static_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        json_lines = [line for line in lines if line.strip() and not line.strip().startswith("//")]
        raw = "".join(json_lines).strip()
        if not raw:
            return 0.0
        if raw.startswith("{"):
            brace_count = 0
            for i, char in enumerate(raw):
                if char == "{": brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        raw = raw[: i + 1]
                        break
        data = json.loads(raw)
        return float(data.get("data_subscription_monthly", 0.0))
    except Exception:
        return 0.0


def calc_commission(quantity, trade_value):
    if quantity <= 0:
        return 0.0
    base = quantity * COMMISSION_PER_SHARE
    comm = max(base, COMMISSION_MINIMUM)
    if trade_value is not None and trade_value > 0:
        comm = min(comm, trade_value * COMMISSION_MAXIMUM_RATE)
    return comm


def token_cost_for_record(record, model_pricing):
    llm = record.get("llm_usage") or {}
    model = (llm.get("model") or record.get("model") or "").strip()
    prices = model_pricing.get(model, {})
    inp = int(llm.get("input_tokens") or 0)
    out = int(llm.get("output_tokens") or 0)
    cache = int(llm.get("cached_tokens") or 0)
    c = (inp / 1000.0) * float(prices.get("input_price_per_k_tokens", 0))
    c += (out / 1000.0) * float(prices.get("output_price_per_k_tokens", 0))
    c += (cache / 1000.0) * float(prices.get("cache_price_per_k_tokens", 0))
    return c


def daily_rows_for_records(
    records,
    model_name,
    frequency_label,
    model_pricing,
    prices_by_date,
    monthly_cost=0.0,
    uncertain_total=0.0,
):
    """
    按记录顺序模拟组合，按日汇总：每日一行 (日期, gross_profit, cost, net_profit)。
    gross_profit = 当日末组合市值 - initial_cash（累计）
    cost = 截至当日末的累计成本，与 fincost 一致：佣金 + token + infra(0.2/条) + monthly(每月首条) + uncertain(可选，首条)
    """
    rows = []
    cash = float(INITIAL_CASH)
    holdings = defaultdict(int)
    last_price = {}
    cumulative_cost = 0.0
    last_day = None
    last_gross = 0.0
    last_cost = 0.0
    last_month = None

    for idx, rec in enumerate(records):
        date_raw = rec.get("date")
        day = get_day(date_raw)
        month_key = (date_raw or "")[:7] if len(str(date_raw or "")) >= 7 else None

        # 与 fincost 一致：每条记录 佣金 + token + infra；每月首条 +monthly；首条 +uncertain
        for t in rec.get("trades", []):
            dt = str(t.get("decision_type") or "").upper()
            ticker = (t.get("ticker") or "").strip()
            qty = int(t.get("quantity") or 0)
            price = t.get("execution_price") or t.get("analysis_price")
            if price is None:
                price = 0.0
            price = float(price)
            if ticker:
                last_price[ticker] = price
            if dt == "BUY":
                cash -= price * qty
                holdings[ticker] += qty
            elif dt == "SELL":
                cash += price * qty
                holdings[ticker] -= qty
            if dt in ("BUY", "SELL"):
                trade_value = price * qty
                cumulative_cost += calc_commission(qty, trade_value)
        cumulative_cost += token_cost_for_record(rec, model_pricing)
        cumulative_cost += INFRA_COST_PER_RECORD
        if month_key and (last_month is None or month_key != last_month):
            cumulative_cost += monthly_cost
        if month_key:
            last_month = month_key
        if idx == 0 and uncertain_total != 0:
            cumulative_cost += uncertain_total

        # 当日末市值：用当日收盘价（prices_by_date[day]）估值，没有则用 last_price
        daily_prices = prices_by_date.get(day, {})
        holdings_value = 0.0
        for ticker, qty in holdings.items():
            if qty == 0:
                continue
            if ticker in daily_prices:
                last_price[ticker] = daily_prices[ticker]
            holdings_value += qty * last_price.get(ticker, 0.0)
        portfolio_value = cash + holdings_value
        gross = portfolio_value - INITIAL_CASH

        # 若进入新的一天，先输出上一天的一行
        if last_day is not None and day != last_day:
            rows.append({
                "模型": model_name,
                "频次": frequency_label,
                "日期": last_day,
                "gross_profit": round(last_gross, 2),
                "cost": round(last_cost, 2),
                "net_profit": round(last_gross - last_cost, 2),
            })
        last_day = day
        last_gross = gross
        last_cost = cumulative_cost

    if last_day is not None:
        rows.append({
            "模型": model_name,
            "频次": frequency_label,
            "日期": last_day,
            "gross_profit": round(last_gross, 2),
            "cost": round(last_cost, 2),
            "net_profit": round(last_gross - last_cost, 2),
        })
    return rows


def get_model_name(records):
    if not records:
        return ""
    llm = records[0].get("llm_usage")
    if isinstance(llm, dict):
        return str(llm.get("model") or "").strip()
    return str(records[0].get("model") or "").strip()


def main():
    model_pricing = load_llm_pricing()
    prices_by_date = load_daily_buy_prices(MERGED_DAILY)
    monthly_exp1 = load_static_monthly(os.path.join(EXP1_DIR, "static-deepseek-v3.2-fast-100000.0-daily-2025-12-01_2026-01-30.jsonl"))
    monthly_exp3 = load_static_monthly(os.path.join(EXP3_DIR, "static-deepseek-v3.2-fast_100000.0_hourly_2025-12-01-10-00-00_2025-12-31-15-00-00.jsonl"))

    exp1_ds = filter_records(
        load_jsonl(
            os.path.join(
                EXP1_DIR,
                "experiment_records_deepseek-v3.2-fast_100000.0_2025-12-01_2026-01-30.jsonl",
            )
        ),
        is_hourly=False,
    )
    exp1_gpt = filter_records(
        load_jsonl(
            os.path.join(
                EXP1_DIR,
                "experiment_records_gpt-5.2_100000_2025-12-01_2026-1-30.jsonl",
            )
        ),
        is_hourly=False,
    )
    exp3_ds = filter_records(
        load_jsonl(
            os.path.join(
                EXP3_DIR,
                "experiment_records_deepseek-v3.2-fast_100000.0_2025-12-01-10-00-00_2025-12-31-15-00-00.jsonl",
            )
        ),
        is_hourly=True,
    )
    exp3_gpt = filter_records(
        load_jsonl(
            os.path.join(
                EXP3_DIR,
                "experiment_records_gpt-5.2_100000.0_2025-12-01-10-00-00_2025-12-31-15-00-00.jsonl",
            )
        ),
        is_hourly=True,
    )

    all_rows = []
    for recs, freq, monthly in [
        (exp1_ds, "daily", monthly_exp1),
        (exp1_gpt, "daily", monthly_exp1),
        (exp3_ds, "hourly", monthly_exp3),
        (exp3_gpt, "hourly", monthly_exp3),
    ]:
        model = get_model_name(recs)
        # 与 fincost 一致：uncertain = 每个交易日 [0, 0.5] 随机数之和；固定 seed 保证可复现
        seed = sum(ord(c) for c in (model + freq)) % (2**32)
        uncertain = compute_uncertain_total(recs, seed=seed)
        all_rows.extend(
            daily_rows_for_records(
                recs, model, freq, model_pricing, prices_by_date,
                monthly_cost=monthly,
                uncertain_total=uncertain,
            )
        )

    # 按 日期、模型、频次 排序，便于阅读
    all_rows.sort(key=lambda r: (r["日期"], r["模型"], r["频次"]))

    # 打印表
    print("时间范围: 2025-12-01 ~ 2025-12-31，每日维度")
    print("字段: 模型, 频次, 日期, gross_profit, cost, net_profit")
    print()
    col = ["模型", "频次", "日期", "gross_profit", "cost", "net_profit"]
    w = [18, 8, 12, 14, 10, 12]
    header = "".join(c.ljust(w[i]) for i, c in enumerate(col))
    print(header)
    print("-" * sum(w))
    for r in all_rows:
        row = "".join(str(r[c]).ljust(w[i]) for i, c in enumerate(col))
        print(row)

    # 保存长表 CSV（原始明细）
    out_path = os.path.join(ROOT, "exp3_daily_comparison.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(col) + "\n")
        for r in all_rows:
            f.write(",".join(str(r[c]) for c in col) + "\n")
    print()
    print(f"已保存: {out_path}")

    # 额外导出 3 个“宽表”：按日期 × (模型-频次)，分别是 gross_profit / net_profit / loss(cost)
    # 行：日期；列：例如 DeepSeek-V3-hourly, DeepSeek-V3-daily, gpt-5.2-hourly, gpt-5.2-daily
    by_day_label = {}
    labels = set()
    dates = set()
    for r in all_rows:
        label = f"{r['模型']}-{r['频次']}"
        day = r["日期"]
        dates.add(day)
        labels.add(label)
        by_day_label.setdefault(day, {})[label] = r
    labels = sorted(labels)
    dates = sorted(dates)

    def write_wide(metric_key: str, filename: str):
        wide_path = os.path.join(ROOT, filename)
        with open(wide_path, "w", encoding="utf-8") as f:
            # 表头：日期 + 各 (模型-频次) 列
            f.write(",".join(["日期"] + labels) + "\n")
            for day in dates:
                row_values = [day]
                row_map = by_day_label.get(day, {})
                for label in labels:
                    cell = ""
                    rec = row_map.get(label)
                    if rec is not None:
                        cell = str(rec.get(metric_key, ""))
                    row_values.append(cell)
                f.write(",".join(row_values) + "\n")
        print(f"已保存: {wide_path}")

    # 这里将 loss 定义为累计成本 cost，便于直接观察“总支出”
    write_wide("gross_profit", "exp3_table_gross_profit.csv")
    write_wide("net_profit", "exp3_table_net_profit.csv")
    write_wide("cost", "exp3_table_loss.csv")


if __name__ == "__main__":
    main()
