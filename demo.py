import os
import sys
import json
import re
from pathlib import Path
import pandas as pd
import random
from datetime import datetime
from datasets import load_dataset

from dotenv import load_dotenv
load_dotenv()


from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
# from langchain_core.callbacks import CallbackManagerForToolCall
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_community.callbacks import get_openai_callback  # For accurate token statistics

# -------------------------- 0. Configuration Loading --------------------------
CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config() -> dict:
    """Load config.json if exists, otherwise return empty dict."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to read config.json, will use default configuration: {e}")
            return {}
    return {}


CONFIG = load_config()

# LLM / API Configuration: Loaded from config.json 
LLM_PROVIDER = CONFIG.get("llm_provider", "openai").lower()
LLM_MODEL = CONFIG.get("llm_model", "gpt-4o-mini")
LLM_BASE_URL = CONFIG.get("llm_base_url")  


def get_llm():
    """
    Select different LLM / API providers based on config file.
    Model configuration (provider and model name) is only loaded from config.json.
    API keys are loaded from environment variables.
    
    Current implementation:
      - LLM_PROVIDER = 'openai'  → Use ChatOpenAI

    Extension point:
      - In the future, other providers (such as Groq, DeepSeek, etc.) can be integrated here,
        just create the corresponding LLM instance based on LLM_PROVIDER branch.
    """
    if LLM_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Missing OPENAI_API_KEY")
            sys.exit(1)
        try:
            llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
            print(f"Using OpenAI model: {LLM_MODEL}")
            return llm
        except Exception as e:
            print(f"Failed to initialize OpenAI LLM: {e}")
            raise
    elif LLM_PROVIDER == "aihubmix":
        api_key = os.getenv("AIHUBMIX_API_KEY")
        if not api_key:
            print("Missing AIHUBMIX_API_KEY")
            sys.exit(1)
        # Get base_url from config.json or environment variable
        base_url = LLM_BASE_URL
        if not base_url:
            print("Missing AIHUBMIX_BASE_URL (required in config.json or .env)")
            sys.exit(1)
        try:
            # Use ChatOpenAI with base_url for OpenAI-compatible APIs
            llm = ChatOpenAI(
                model=LLM_MODEL,
                temperature=0,
                base_url=base_url,
                api_key=api_key
            )
            print(f"Using Aihubmix model: {LLM_MODEL} at {base_url}")
            return llm
        except Exception as e:
            print(f"Failed to initialize Aihubmix LLM: {e}")
            raise
    else:
        # Placeholder for other providers, for future extension
        raise NotImplementedError(f"LLM_PROVIDER not implemented: {LLM_PROVIDER}")

# -------------------------- 1. Data Loading --------------------------
print("[Step 1] Loading dataset from Hugging Face...")
ds = load_dataset("pavement/tsla_stock_price")
print("[Step 1] Dataset loaded successfully.")

# Handle DatasetDict - get the 'train' split (or first available split)
if hasattr(ds, 'keys'):
    print(f"[Step 1] Dataset has splits: {list(ds.keys())}")
    # Use 'train' split if available, otherwise use the first split
    split_name = 'train' if 'train' in ds.keys() else list(ds.keys())[0]
    dataset = ds[split_name]
    print(f"[Step 1] Using split: {split_name}")
else:
    dataset = ds

# Convert to DataFrame using .to_pandas() method for HuggingFace Dataset
print("[Step 1] Converting to DataFrame...")
df = dataset.to_pandas()
print(f"[Step 1] DataFrame columns: {list(df.columns)}")
print(f"[Step 1] DataFrame shape: {df.shape}")

# Verify 'start' column exists
if 'start' not in df.columns:
    print(f"[ERROR] 'start' column not found. Available columns: {list(df.columns)}")
    raise KeyError("'start' column not found in dataset")

# Process date column
df['date'] = pd.to_datetime(df['start'])
df = df.sort_values('date').reset_index(drop=True)

print(f"[Step 1] Data loaded: {len(df)} rows, Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")

# Current row target[0] = closing price of the day (decision information)
# Next row target[0] = next day closing price ≈ next day opening execution price

# -------------------------- 2. Commission Calculation Function --------------------------
def calculate_commission(shares: int, trade_value: float) -> float:
    """
    Calculate commission based on Interactive Brokers Fixed pricing structure.
    Reference: https://www.interactivebrokers.com/cn/pricing/commissions-stocks.php
    
    IB Fixed Pricing:
    - Per share: $0.005 per share
    - Minimum: $1.00 per order
    - Maximum: 1% of trade value
    
    Args:
        shares: Number of shares traded
        trade_value: Total value of the trade (price * shares)
    
    Returns:
        Commission amount in USD
    """
    if shares == 0 or trade_value == 0:
        return 0.0
    
    # Calculate base commission: per share * number of shares
    base_commission = shares * COMMISSION_PER_SHARE
    
    # Apply minimum commission
    commission = max(base_commission, COMMISSION_MINIMUM)
    
    # Apply maximum commission (percentage of trade value)
    max_commission = trade_value * COMMISSION_MAXIMUM_RATE
    commission = min(commission, max_commission)
    
    return commission

# -------------------------- 3. Tools --------------------------
@tool
def get_current_price(date: str) -> float:
    """Get TSLA closing price (target[0]) for the specified date"""
    row = df[df['start'] == date]
    if len(row) == 0:
        return None
    return row.iloc[0]['target'][0]

@tool
def get_historical_prices(days: int = 10) -> str:
    """Get closing price sequence for the last 'days' days (for Agent to reference trends)"""
    if len(df) < days:
        return str(df['target'].apply(lambda x: x[0]).tolist())
    prices = df['target'].apply(lambda x: x[0]).tail(days).tolist()
    return str(prices)

tools = [get_current_price, get_historical_prices]

# -------------------------- 4. LLM & Agent --------------------------
print("[Step 2] Initializing LLM...")
llm = get_llm()
print("[Step 2] LLM initialized successfully.")

print("[Step 3] Creating agent...")
# Create agent using create_agent (new API does not require AgentExecutor)
agent = create_agent(llm, tools=tools)
print("[Step 3] Agent created successfully.")

# -------------------------- 5. Backtest Parameters --------------------------
initial_cash = float(CONFIG.get("initial_cash", 10000.0))  # Initial capital (loaded from config.json)

# Start date configuration (from config.json or default to first date in dataset)
start_date_str = CONFIG.get("start_date")  # Format: "YYYY-MM-DD" or None to start from beginning
if start_date_str:
    # Find the index of the start date
    start_date_idx = df[df['start'] == start_date_str].index
    if len(start_date_idx) > 0:
        start_idx = start_date_idx[0]
        print(f"[Config] Starting from date: {start_date_str} (index: {start_idx})")
    else:
        print(f"[Warning] Start date {start_date_str} not found in dataset. Starting from beginning.")
        start_idx = 0
else:
    start_idx = 0
    print(f"[Config] No start_date specified, starting from first date in dataset: {df.iloc[0]['start']}")

# Adjust dataframe to start from specified date
df = df.iloc[start_idx:].reset_index(drop=True)
max_days = len(df) - 1          # Maximum trading days

# Cost parameters (loaded from config.json)
# Interactive Brokers commission structure (Fixed pricing)
# Reference: https://www.interactivebrokers.com/cn/pricing/commissions-stocks.php
COMMISSION_TYPE = CONFIG.get("commission_type", "fixed")  # "fixed" or "tiered"
COMMISSION_PER_SHARE = float(CONFIG.get("commission_per_share", 0.005))  # Per share commission USD (IB Fixed: $0.005/share)
COMMISSION_MINIMUM = float(CONFIG.get("commission_minimum", 1.0))  # Minimum commission per order USD (IB Fixed: $1.00)
COMMISSION_MAXIMUM_RATE = float(CONFIG.get("commission_maximum_rate", 0.01))  # Maximum commission as % of trade value (IB: 1%)
INFRA_DAILY = float(CONFIG.get("infra_daily", 0.5))  # Daily static infrastructure cost USD
RANDOM_COST_DAILY_RANGE = (
    float(CONFIG.get("random_cost_min", 0.0)),
    float(CONFIG.get("random_cost_max", 1.0)),
)  # Uncertainty cost range

# LLM costs (loaded from config.json)
# Default values are official prices for gpt-4o-mini
INPUT_PRICE_PER_K_TOKENS = float(CONFIG.get("input_price_per_k_tokens", 0.00015 / 1000))
OUTPUT_PRICE_PER_K_TOKENS = float(CONFIG.get("output_price_per_k_tokens", 0.00060 / 1000))

# Data subscription cost (monthly)
DATA_SUBSCRIPTION_MONTHLY = float(CONFIG.get("data_subscription_monthly", 0.0))  # Monthly data subscription cost USD

# -------------------------- 6. Backtest Main Loop --------------------------
print(f"[Step 4] Starting backtest with {max_days} days...")
print(f"[Step 4] Initial cash: ${initial_cash:.2f}")
records = []
llm_outputs_records = []  # Separate records for LLM outputs

cash = initial_cash
shares = 0
cumulative_cost = 0.0
cumulative_gross_profit = 0.0  # Gross profit without deducting costs (position + cash changes)

# Track current month for monthly cost calculation
current_month = None

for i in range(max_days):
    if i % 10 == 0:  # Print progress every 10 days
        print(f"[Progress] Day {i+1}/{max_days}")
    current_row = df.iloc[i]
    date_str = current_row['start']
    current_price = current_row['target'][0]  # Day t closing price → visible to Agent
    
    # Parse date to detect month changes
    current_date = pd.to_datetime(date_str)
    current_date_month = (current_date.year, current_date.month)  # Use (year, month) tuple
    
    # Check if month has changed
    monthly_subscription_cost_today = 0.0
    if current_month is None:
        # First day: initialize and add monthly cost
        current_month = current_date_month
        monthly_subscription_cost_today = DATA_SUBSCRIPTION_MONTHLY
    elif current_date_month != current_month:
        # Month changed: add monthly cost for the new month
        current_month = current_date_month
        monthly_subscription_cost_today = DATA_SUBSCRIPTION_MONTHLY

    # Get historical prices for Agent
    history_prices = df.iloc[max(0, i-9):i+1]['target'].apply(lambda x: x[0]).tolist()

    # Agent decision + record actual token consumption
    try:
        with get_openai_callback() as cb:
            # Use new API messages format
            messages = [
                SystemMessage(content="""
You are a trading agent analyzing market data to make investment decisions about TSLA stock.
Based on your analysis, provide a specific recommendation to buy, sell, or hold. 

IMPORTANT: You will receive the current trading date in each request. Pay attention to the date as it helps you understand the temporal context of the market data.

Do not forget to utilize lessons from past decisions to learn from your mistakes. 
Here are some reflections from similar situations you traded in and the lessons learned:
- Consider market trends and volatility patterns
- Balance risk and reward based on available capital
- Avoid emotional trading decisions
- Learn from previous gains and losses

IMPORTANT: You must provide your decision reasoning BEFORE stating your decision.

Output format:
1. First, explain your analysis and reasoning for the decision (decision rationale)
2. Then, conclude with your decision in the exact format:
   - For buy: "buy <number>" where <number> is the number of shares to buy (positive integer)
   - For sell: "sell <number>" where <number> is the number of shares to sell (positive integer)
   - For hold: "hold" (no number needed)

Examples:
- "Based on the price trend showing a 10% increase over the last week and current momentum, I recommend buying. buy 100"
- "The stock price has been declining for 5 consecutive days with high volatility, indicating potential further decline. sell 50"
- "The price is relatively stable with no clear trend, and I want to wait for more signals. hold"

Always provide your reasoning before the decision statement.
"""),
                HumanMessage(content=f"Trading Date: {date_str}\nLast 10 days closing price sequence: {history_prices}\nToday's closing price: {current_price:.3f}\nAvailable cash: ${cash:.2f}\nCurrent shares: {shares}\n\nPlease analyze the market data and provide your investment decision in the format: buy <number>, sell <number>, or hold:")
            ]
            if i == 0:  # Only print for first call to avoid spam
                print(f"[API] Calling agent for day {i+1} (date: {date_str})...")
            result = agent.invoke({"messages": messages})
            if i == 0:  # Only print for first call to avoid spam
                print(f"[API] Agent response received.")
            
            # Extract LLM output - save complete original output (last AIMessage only)
            decision_raw = ""
            llm_output_raw = ""  # Store complete original LLM output (last AIMessage content)
            
            if result.get("messages"):
                # Find the last AI message from the end (not ToolMessage)
                last_ai_message = None
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        last_ai_message = msg
                        break
                
                if last_ai_message:
                    # Save the complete original content (without lowercasing)
                    llm_output_raw = last_ai_message.content if last_ai_message.content else ""
                    # For parsing decision, use lowercase version
                    decision_raw = llm_output_raw.strip().lower()
                else:
                    # Fallback: use the last message
                    last_message = result["messages"][-1]
                    if hasattr(last_message, 'content'):
                        llm_output_raw = last_message.content if last_message.content else str(last_message)
                    else:
                        llm_output_raw = str(last_message)
                    decision_raw = llm_output_raw.strip().lower()
            else:
                llm_output_raw = str(result)
                decision_raw = llm_output_raw.strip().lower()
            
            input_tokens = cb.prompt_tokens
            output_tokens = cb.completion_tokens
            llm_cost = (input_tokens * INPUT_PRICE_PER_K_TOKENS +
                        output_tokens * OUTPUT_PRICE_PER_K_TOKENS)
    except Exception as e:
        print(f"[ERROR] Failed to get agent decision at day {i+1} (date: {date_str}): {e}")
        raise

    # Parse decision - extract action and quantity, and extract reasoning
    # Format: reasoning text, then "buy <number>", "sell <number>", or "hold"
    action = "hold"
    quantity = 0

    
    # Try to parse "buy <number>" or "sell <number>" format (strict matching)
    buy_match = re.search(r'\bbuy\s+(\d+)\b', decision_raw)
    sell_match = re.search(r'\bsell\s+(\d+)\b', decision_raw)
    
    if buy_match:
        # Extract quantity and validate it's positive
        quantity = int(buy_match.group(1))
        if quantity > 0:
            action = "buy"
        else:
            action = "hold"  # Invalid: quantity must be positive
            quantity = 0
    elif sell_match:
        # Extract quantity and validate it's positive
        quantity = int(sell_match.group(1))
        if quantity > 0:
            action = "sell"
        else:
            action = "hold"  # Invalid: quantity must be positive
            quantity = 0
    elif re.search(r'\bhold\b', decision_raw) or not decision_raw.strip():
        # Explicit hold or empty response
        action = "hold"
        quantity = 0
    else:
        # Format not recognized - default to hold (strict mode: no fallback to default quantity)
        action = "hold"
        quantity = 0
        # Optionally log or print warning about unrecognized format
        if "buy" in decision_raw.lower() or "sell" in decision_raw.lower():
            print(f"Warning: Unrecognized format at {date_str}. Expected 'buy <number>', 'sell <number>', or 'hold'. Got: {decision_raw[:100]}")

    # Next day execution price (more accurate: next row target[0])
    if i + 1 >= len(df):
        break
    exec_price = df.iloc[i + 1]['target'][0]

    # Execute trade
    trade_count = 0
    trade_value = 0.0
    slippage = 0.0
    actual_quantity = 0  # Actual quantity executed

    if action == "buy" and quantity > 0:
        # Calculate maximum affordable shares
        max_affordable = int(cash / exec_price)
        actual_quantity = min(quantity, max_affordable)  # Can't buy more than affordable
        
        if actual_quantity > 0:
            trade_value = exec_price * actual_quantity
            shares += actual_quantity
            cash -= trade_value
            trade_count = 1
            # Slippage: difference between execution price and decision price, multiplied by actual quantity
            slippage = (exec_price - current_price) * actual_quantity if exec_price > current_price else 0

    elif action == "sell" and quantity > 0:
        # Can't sell more than owned
        actual_quantity = min(quantity, shares)
        
        if actual_quantity > 0:
            trade_value = exec_price * actual_quantity
            shares -= actual_quantity
            cash += trade_value
            trade_count = 1
            # Slippage: difference between decision price and execution price, multiplied by actual quantity
            slippage = (current_price - exec_price) * actual_quantity if exec_price < current_price else 0

    # For hold action, no changes (trade_count=0, trade_value=0, slippage=0 already set)

    # Trading costs (Interactive Brokers commission structure)
    # IB charges commission per share with minimum and maximum limits
    # Spread rate is not part of IB commission structure (removed)
    if trade_count > 0 and actual_quantity > 0:
        commission = calculate_commission(actual_quantity, trade_value)
    else:
        commission = 0.0
    trading_cost_today = commission + slippage

    # Other costs
    infra_cost = INFRA_DAILY
    random_cost = random.uniform(*RANDOM_COST_DAILY_RANGE)
    # Monthly subscription cost (added only on month change, otherwise 0)
    daily_total_cost = trading_cost_today + llm_cost + infra_cost + random_cost + monthly_subscription_cost_today

    cumulative_cost += daily_total_cost

    # Daily market value (valued at execution price, more fair)
    market_value = cash + shares * exec_price
    cumulative_gross_profit = market_value - initial_cash  # Gross profit
    cumulative_net_profit = cumulative_gross_profit - cumulative_cost

    # Record
    records.append({
        "date": date_str,
        "hold_shares": shares,
        "decision": action,
        "requested_quantity": quantity,  # Agent requested quantity
        "actual_quantity": actual_quantity,  # Actually executed quantity
        "market_value": round(market_value, 2),
        "daily_gross_pnl": round(cumulative_gross_profit - (records[-1]['market_value'] + records[-1]['daily_gross_pnl'] if records else initial_cash), 2),
        "trade_count": trade_count,
        "daily_trading_cost": round(trading_cost_today, 4),
        "daily_llm_input_tokens": input_tokens,
        "daily_llm_output_tokens": output_tokens,
        "daily_llm_cost_usd": round(llm_cost, 6),
        "daily_infra_cost": infra_cost,
        "daily_random_cost": round(random_cost, 4),
        "monthly_data_subscription_cost": round(monthly_subscription_cost_today, 4),
        "daily_total_cost": round(daily_total_cost, 4),
        "cumulative_cost": round(cumulative_cost, 2),
        "cumulative_net_profit": round(cumulative_net_profit, 2),
        "slippage_usd": round(slippage, 2)
    })
    
    # Record LLM output separately
    llm_outputs_records.append({
        "date": date_str,
        "day": i + 1,
        "current_price": round(current_price, 3),
        "available_cash": round(cash, 2),
        "current_shares": shares,
        "decision": action,
        "requested_quantity": quantity,
        "actual_quantity": actual_quantity,
        "llm_output": llm_output_raw,  # Full original LLM output
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "llm_cost_usd": round(llm_cost, 6)
    })

    # Termination judgment
    # if market_value < daily_total_cost * 30:  # Expected unable to cover future 1 month costs
    #     print(f"[Bankruptcy Termination] Date: {date_str}, Remaining market value: ${market_value:.2f}, Ran for {i+1} days")
    #     break

    if market_value < daily_total_cost:
        print(f"[Bankruptcy Termination] Date: {date_str}, Remaining market value: ${market_value:.2f}, Ran for {i+1} days")
        break

    # if i > 60:  # Run at least 60 days before judging balance
    #     recent_records = pd.DataFrame(records[-30:])
    #     avg_daily_cost = recent_records['daily_total_cost'].mean()
    #     avg_daily_net = recent_records['cumulative_net_profit'].diff().mean()
    #     if cumulative_net_profit >= 0 and avg_daily_net > avg_daily_cost * 1.1:
    #         print(f"[Balance Achieved] Date: {date_str}, Time taken: {i+1} days, Cumulative net profit: ${cumulative_net_profit:.2f}")
    #         break

    recent_records = pd.DataFrame(records[-30:])
    avg_daily_cost = recent_records['daily_total_cost'].mean()
    avg_daily_net = recent_records['cumulative_net_profit'].diff().mean()
    if cumulative_net_profit >= 0 and avg_daily_net > avg_daily_cost * 1.1:
        print(f"[Balance Achieved] Date: {date_str}, Time taken: {i+1} days, Cumulative net profit: ${cumulative_net_profit:.2f}")
        break

# -------------------------- 6. Save Results --------------------------
result_df = pd.DataFrame(records)
result_df.to_csv("tsla_agent_real_profit_backtest.csv", index=False)

# Save LLM outputs to separate CSV
llm_outputs_df = pd.DataFrame(llm_outputs_records)
llm_outputs_df.to_csv("tsla_agent_llm_outputs.csv", index=False)

print("\nBacktest completed! Results saved:")
print(f"  - Main results: tsla_agent_real_profit_backtest.csv")
print(f"  - LLM outputs: tsla_agent_llm_outputs.csv")
print(f"Total trading days: {len(result_df)}")
print(f"Final market value: ${result_df.iloc[-1]['market_value']:.2f}")
print(f"Cumulative total cost: ${result_df.iloc[-1]['cumulative_cost']:.2f}")
print(f"Cumulative net profit: ${result_df.iloc[-1]['cumulative_net_profit']:.2f}")

# Display last 10 days
print("\nLast 10 days records:")
print(result_df[[
    'date', 'hold_shares', 'decision', 'market_value',
    'daily_total_cost', 'daily_llm_cost_usd',
    'cumulative_cost', 'cumulative_net_profit'
]].tail(10))