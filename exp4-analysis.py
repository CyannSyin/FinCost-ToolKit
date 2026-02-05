"""
exp4是跨周期的实验
data/KDD/exp-4 这里是2个模型3个周期的实验结果，要进行对比，输出的结果是表格
只需要读取experiment_records这个开头的数据，计算
model，周期（一共三个），gross profit， cost，然后不同的cost分布， net profit这样一个表
cost分布：commission_total, token_total, infra_total, monthly_total, uncertain_cost, total_cost
"""
import json
import os
import re
import random
from collections import defaultdict
from datetime import datetime, timedelta

# 项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
EXP4_DIR = os.path.join(DATA_DIR, "KDD", "exp-4")
CONFIG_LLM = os.path.join(ROOT, "config_llm.json")
MERGED_DAILY = os.path.join(DATA_DIR, "exp4_data.jsonl")

INITIAL_CASH = 100000.0

# 佣金参数（与 fincost/commission 一致）
COMMISSION_PER_SHARE = 0.005
COMMISSION_MINIMUM = 1.0
COMMISSION_MAXIMUM_RATE = 0.01

# 与 fincost/main 一致：infra 按每条记录 0.2；monthly 来自 static 的 data_subscription_monthly；uncertain 可选
INFRA_COST_PER_RECORD = 0.2


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # 跳过注释行
            if line.startswith("//"):
                continue
            try:
                # 修复常见的 JSON 格式问题
                # 1. 移除数字后的尾随空格（在逗号或}之前）
                line = re.sub(r'(\d+)\s+([,}])', r'\1\2', line)
                # 2. 修复错误的格式：{"trade": [...]} 或 {"tradeS": [...]} -> "trades": [...]
                line = re.sub(r'"tool_usage":\s*\{[^}]*\},\s*\{\s*"trade[sS]?":\s*', r'"tool_usage": {"calls": []}, "trades": ', line)
                # 3. 如果解析失败，尝试只解析第一个完整的 JSON 对象
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # 尝试只解析第一个完整的 JSON 对象
                    decoder = json.JSONDecoder()
                    obj, idx = decoder.raw_decode(line)
                    records.append(obj)
            except json.JSONDecodeError as e:
                print(f"JSON decode error in {path}, line {line_num}: {e}")
                print(f"Problematic line (first 400 chars): {line[:400]}")
                raise
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


def load_llm_pricing(path=CONFIG_LLM):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("models", {})


def load_static_monthly(static_path):
    """从 static 文件读 data_subscription_monthly（与 fincost 一致）"""
    try:
        with open(static_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 移除注释行和空行
        json_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                json_lines.append(line)
        raw = "".join(json_lines).strip()
        if not raw:
            return 0.0
        # 如果是 JSONL 格式，只解析第一行
        if raw.startswith("{"):
            brace_count = 0
            json_end = 0
            for i, char in enumerate(raw):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            if json_end > 0:
                raw = raw[:json_end]
        data = json.loads(raw)
        return float(data.get("data_subscription_monthly", 0.0))
    except Exception as e:
        print(f"Warning: Failed to load monthly from {static_path}: {e}")
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


def _use_two_monthly_rule(period_start, period_end):
    """2-4月、6-8月按 2 个月扣费：首日扣一次，隔 30 天再扣一次。"""
    if not period_start or not period_end:
        return False
    try:
        start = datetime.strptime(period_start[:10], "%Y-%m-%d")
        end = datetime.strptime(period_end[:10], "%Y-%m-%d")
        sm, em = start.month, end.month
        return (sm == 2 and em == 4) or (sm == 6 and em == 8)
    except Exception:
        return False


def analyze_period(records, model_pricing, prices_by_date, monthly_cost, uncertain_total=0.0,
                   period_start=None, period_end=None):
    """
    分析一个周期的所有记录，返回汇总结果：
    - gross_profit: 最终组合市值 - initial_cash
    - cost 分解：commission_total, token_total, infra_total, monthly_total, uncertain_cost, total_cost
    - net_profit: gross_profit - total_cost
    - 2-4月、6-8月：monthly 按 2 个月扣，交易开始扣一次，隔日历 30 天再扣一次；其余周期按自然月扣。
    """
    cash = float(INITIAL_CASH)
    holdings = defaultdict(int)
    last_price = {}
    
    # 成本分项累计
    commission_total = 0.0
    token_total = 0.0
    infra_total = 0.0
    monthly_total = 0.0
    
    last_month = None
    use_two_monthly = _use_two_monthly_rule(period_start or "", period_end or "")
    # 2-4月/6-8月：首日+30 天后再扣一次
    second_monthly_added = False
    start_plus_30 = None
    if use_two_monthly and period_start:
        try:
            start_d = datetime.strptime(period_start[:10], "%Y-%m-%d")
            start_plus_30 = (start_d + timedelta(days=30)).strftime("%Y-%m-%d")
        except Exception:
            start_plus_30 = None
    
    # 处理所有记录
    for idx, rec in enumerate(records):
        date_raw = rec.get("date")
        date_str = (date_raw or "")[:10] if len(str(date_raw or "")) >= 10 else ""
        month_key = (date_raw or "")[:7] if len(str(date_raw or "")) >= 7 else None
        
        # 应用交易
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
                commission_total += calc_commission(qty, trade_value)
        
        # Token 成本
        token_total += token_cost_for_record(rec, model_pricing)
        
        # Infra 成本（每条记录）
        infra_total += INFRA_COST_PER_RECORD
        
        # Monthly 成本
        if use_two_monthly and start_plus_30:
            # 2-4月/6-8月：首条扣一次，首条日期>=start+30 再扣一次
            if idx == 0:
                monthly_total += monthly_cost
            if not second_monthly_added and date_str >= start_plus_30:
                monthly_total += monthly_cost
                second_monthly_added = True
        else:
            # 按自然月：每月首条扣一次
            if month_key and (last_month is None or month_key != last_month):
                monthly_total += monthly_cost
            if month_key:
                last_month = month_key
    
    # Uncertain 成本（整笔加在第一条）
    if uncertain_total != 0 and records:
        uncertain_cost = uncertain_total
    else:
        # 如果没有指定，按 fincost 方式：trading_days 个随机数求和
        trading_days = len(set(rec.get("date", "")[:10] for rec in records if rec.get("date")))
        uncertain_cost = sum(random.uniform(0.0, 0.5) for _ in range(trading_days))
    
    total_cost = commission_total + token_total + infra_total + monthly_total + uncertain_cost
    
    # 计算最终组合市值（用最后一条记录的日期）
    if records:
        last_record_date = records[-1].get("date", "")
        last_day = last_record_date[:10] if " " in str(last_record_date) or "T" in str(last_record_date) else str(last_record_date)
        daily_prices = prices_by_date.get(last_day, {})
        holdings_value = 0.0
        for ticker, qty in holdings.items():
            if qty == 0:
                continue
            if ticker in daily_prices:
                last_price[ticker] = daily_prices[ticker]
            holdings_value += qty * last_price.get(ticker, 0.0)
        portfolio_value = cash + holdings_value
        gross_profit = portfolio_value - INITIAL_CASH
    else:
        gross_profit = 0.0
    
    net_profit = gross_profit - total_cost
    
    return {
        "gross_profit": round(gross_profit, 2),
        "commission_total": round(commission_total, 2),
        "token_total": round(token_total, 2),
        "infra_total": round(infra_total, 2),
        "monthly_total": round(monthly_total, 2),
        "uncertain_cost": round(uncertain_cost, 2),
        "total_cost": round(total_cost, 2),
        "net_profit": round(net_profit, 2),
    }


def get_model_name(records):
    if not records:
        return ""
    llm = records[0].get("llm_usage")
    if isinstance(llm, dict):
        return str(llm.get("model") or "").strip()
    return str(records[0].get("model") or "").strip()


def extract_period_from_filename(filename):
    """从文件名提取周期标识，如 '2025-02-10_2025-04-07' -> '2025-02-10 to 2025-04-07'"""
    # 匹配日期范围模式，支持单数字月份/日期（如 2026-1-30）
    import re
    # 匹配 YYYY-MM-DD 或 YYYY-M-D 格式
    match = re.search(r'(\d{4}-\d{1,2}-\d{1,2})[_-](\d{4}-\d{1,2}-\d{1,2})', filename)
    if match:
        start, end = match.groups()
        # 标准化日期格式（补零）
        def normalize_date(d):
            parts = d.split('-')
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        start_norm = normalize_date(start)
        end_norm = normalize_date(end)
        return f"{start_norm} to {end_norm}"
    return filename


def main():
    model_pricing = load_llm_pricing()
    prices_by_date = load_daily_buy_prices(MERGED_DAILY)
    
    # 扫描 exp-4 目录，找到所有 experiment_records 开头的文件
    experiment_files = []
    for filename in os.listdir(EXP4_DIR):
        if filename.startswith("experiment_records_") and filename.endswith(".jsonl"):
            filepath = os.path.join(EXP4_DIR, filename)
            experiment_files.append((filename, filepath))
    
    # 按模型和周期分组
    results = []
    
    for filename, filepath in sorted(experiment_files):
        # 提取模型名和周期
        # 格式：experiment_records_<model>_<initial_cash>_<start>_<end>.jsonl
        parts = filename.replace("experiment_records_", "").replace(".jsonl", "").split("_")
        if len(parts) < 4:
            print(f"Warning: Cannot parse filename {filename}, skipping")
            continue
        
        # 模型名可能是 deepseek-v3.2-fast 或 gpt-5.2
        # 初始资金是 100000 或 100000.0
        # 日期范围是最后两个部分
        model_name = parts[0]
        if len(parts) > 4:
            # 模型名可能包含下划线，尝试合并
            for i in range(1, len(parts) - 3):
                if parts[i].startswith("100000"):
                    model_name = "_".join(parts[:i])
                    break
        
        period = extract_period_from_filename(filename)
        
        # 加载记录
        records = load_jsonl(filepath)
        if not records:
            print(f"Warning: No records in {filename}, skipping")
            continue
        
        # 找到对应的 static 文件
        static_pattern = f"static-{model_name}-100000.0-daily-{period.replace(' to ', '_')}.jsonl"
        static_path = os.path.join(EXP4_DIR, static_pattern)
        if not os.path.exists(static_path):
            # 尝试其他可能的格式
            static_pattern2 = f"static-{model_name}-100000-daily-{period.replace(' to ', '_')}.jsonl"
            static_path = os.path.join(EXP4_DIR, static_pattern2)
        if not os.path.exists(static_path):
            # 尝试 gpt-5.2 的特殊格式
            static_pattern3 = f"static-gpt-5.2-100000.0-daily-{period.replace(' to ', '_')}.jsonl"
            static_path = os.path.join(EXP4_DIR, static_pattern3)
        
        monthly_cost = load_static_monthly(static_path) if os.path.exists(static_path) else 100.0
        
        # 解析周期起止（2-4月/6-8月 按 2 个月扣费：首日+隔 30 天再扣）
        period_start, period_end = None, None
        if " to " in period:
            parts = period.split(" to ", 1)
            if len(parts) == 2:
                period_start, period_end = parts[0].strip(), parts[1].strip()
        
        # 分析这个周期
        analysis = analyze_period(
            records, model_pricing, prices_by_date, monthly_cost, uncertain_total=0.0,
            period_start=period_start, period_end=period_end,
        )
        
        results.append({
            "model": model_name,
            "period": period,
            **analysis,
        })
    
    # 按模型和周期排序
    results.sort(key=lambda r: (r["model"], r["period"]))
    
    # 打印表格
    print("=" * 150)
    print("exp4 跨周期对比表")
    print("=" * 150)
    cols = ["model", "period", "gross_profit", "commission_total", "token_total", "infra_total", 
            "monthly_total", "uncertain_cost", "total_cost", "net_profit"]
    widths = [22, 38, 14, 18, 14, 14, 14, 18, 12, 12]
    
    # 表头，每列之间加空格
    header_parts = []
    for c, w in zip(cols, widths):
        header_parts.append(c.ljust(w))
    print("  ".join(header_parts))
    print("-" * (sum(widths) + 2 * (len(cols) - 1)))
    
    for r in results:
        row_parts = []
        for c, w in zip(cols, widths):
            val = str(r.get(c, ""))
            row_parts.append(val.ljust(w))
        print("  ".join(row_parts))
    
    print("=" * 120)
    
    # 保存 CSV
    csv_path = os.path.join(ROOT, "exp4_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
