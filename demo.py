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
# New dataset: TheFinAI/CLEF_Task3_Trading
# Directly load TSLA split only (more efficient - only downloads TSLA data)
dataset = load_dataset("TheFinAI/CLEF_Task3_Trading", split="TSLA")
print("[Step 1] Dataset loaded successfully.")
print(f"[Step 1] Loaded TSLA split with {len(dataset)} rows")

# ========== OLD DATASET LOADING CODE (COMMENTED OUT) ==========
# # Load entire dataset first, then select split
# ds = load_dataset("TheFinAI/CLEF_Task3_Trading")
# print("[Step 1] Dataset loaded successfully.")
# 
# # Handle DatasetDict - get the 'TSLA' split
# if hasattr(ds, 'keys'):
#     print(f"[Step 1] Dataset has splits: {list(ds.keys())}")
#     # Use 'TSLA' split for TSLA stock data
#     if 'TSLA' in ds.keys():
#         split_name = 'TSLA'
#         dataset = ds[split_name]
#         print(f"[Step 1] Using split: {split_name}")
#     else:
#         print(f"[ERROR] TSLA split not found. Available splits: {list(ds.keys())}")
#         raise KeyError("TSLA split not found in dataset")
# else:
#     dataset = ds
# ========== END OLD DATASET LOADING CODE ==========

# Convert to DataFrame using .to_pandas() method for HuggingFace Dataset
print("[Step 1] Converting to DataFrame...")
df = dataset.to_pandas()
print(f"[Step 1] DataFrame columns: {list(df.columns)}")
print(f"[Step 1] DataFrame shape: {df.shape}")

# Verify required columns exist
required_columns = ['date', 'prices']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"[ERROR] Required columns not found: {missing_columns}. Available columns: {list(df.columns)}")
    raise KeyError(f"Required columns not found: {missing_columns}")

# Check if news column exists
if 'news' not in df.columns:
    print(f"[Warning] 'news' column not found in dataset. Available columns: {list(df.columns)}")
    print(f"[Warning] News information will not be available during backtest.")
else:
    # Check news column data types and sample values
    news_sample_count = df['news'].notna().sum()
    print(f"[Step 1] News column found: {news_sample_count}/{len(df)} rows have news data")
    if news_sample_count > 0:
        sample_news = df[df['news'].notna()].iloc[0]['news']
        print(f"[Step 1] Sample news type: {type(sample_news)}")
        if isinstance(sample_news, list):
            print(f"[Step 1] Sample news is list with {len(sample_news)} items")
        elif isinstance(sample_news, str):
            print(f"[Step 1] Sample news is string with length {len(sample_news)}")

# Process date column (already exists as 'date' in new dataset)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Add 'start' column for compatibility with existing code (using date as string)
df['start'] = df['date'].dt.strftime('%Y-%m-%d')

print(f"[Step 1] Data loaded: {len(df)} rows, Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")

# Current row prices = closing price of the day (decision information)
# Next row prices = next day closing price ≈ next day opening execution price

# ========== OLD DATASET CODE (COMMENTED OUT) ==========
# # Old dataset: pavement/tsla_stock_price
# ds = load_dataset("pavement/tsla_stock_price")
# print("[Step 1] Dataset loaded successfully.")
# 
# # Handle DatasetDict - get the 'train' split (or first available split)
# if hasattr(ds, 'keys'):
#     print(f"[Step 1] Dataset has splits: {list(ds.keys())}")
#     # Use 'train' split if available, otherwise use the first split
#     split_name = 'train' if 'train' in ds.keys() else list(ds.keys())[0]
#     dataset = ds[split_name]
#     print(f"[Step 1] Using split: {split_name}")
# else:
#     dataset = ds
# 
# # Convert to DataFrame using .to_pandas() method for HuggingFace Dataset
# print("[Step 1] Converting to DataFrame...")
# df = dataset.to_pandas()
# print(f"[Step 1] DataFrame columns: {list(df.columns)}")
# print(f"[Step 1] DataFrame shape: {df.shape}")
# 
# # Verify 'start' column exists
# if 'start' not in df.columns:
#     print(f"[ERROR] 'start' column not found. Available columns: {list(df.columns)}")
#     raise KeyError("'start' column not found in dataset")
# 
# # Process date column
# df['date'] = pd.to_datetime(df['start'])
# df = df.sort_values('date').reset_index(drop=True)
# 
# print(f"[Step 1] Data loaded: {len(df)} rows, Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")
# 
# # Current row target[0] = closing price of the day (decision information)
# # Next row target[0] = next day closing price ≈ next day opening execution price
# ========== END OLD DATASET CODE ==========

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
    """Get TSLA closing price (prices) for the specified date"""
    row = df[df['start'] == date]
    if len(row) == 0:
        return None
    return float(row.iloc[0]['prices'])

@tool
def get_historical_prices(days: int = 10) -> str:
    """Get closing price sequence for the last 'days' days (for Agent to reference trends)"""
    if len(df) < days:
        return str(df['prices'].tolist())
    prices = df['prices'].tail(days).tolist()
    return str(prices)

@tool
def get_news(date: str) -> str:
    """Get TSLA news articles for the specified date. Returns news content that may affect stock price.
    
    Args:
        date: Date in format 'YYYY-MM-DD' (e.g., '2024-11-07')
    
    Returns:
        String containing news articles for the specified date, or error message if not found.
    """
    row = df[df['start'] == date]
    if len(row) == 0:
        return f"No data found for date: {date}"
    
    try:
        news_list = row.iloc[0]['news']
        
        # Handle different possible formats
        if news_list is None:
            return f"No news available for date: {date}"
        
        if isinstance(news_list, list):
            if len(news_list) == 0:
                return f"No news available for date: {date}"
            # If it's a list of strings, join them with separators
            if len(news_list) > 0 and isinstance(news_list[0], str):
                # Join multiple news articles with clear separators
                return "\n\n" + "="*80 + "\n\n".join([f"News Article {i+1}:\n{article}" for i, article in enumerate(news_list)])
            else:
                return str(news_list)
        elif isinstance(news_list, str):
            # If it's already a string, return as is
            return news_list
        else:
            return str(news_list)
    except Exception as e:
        return f"Error retrieving news for date {date}: {str(e)}"

# ========== OLD TOOLS CODE (COMMENTED OUT) ==========
# @tool
# def get_current_price(date: str) -> float:
#     """Get TSLA closing price (target[0]) for the specified date"""
#     row = df[df['start'] == date]
#     if len(row) == 0:
#         return None
#     return row.iloc[0]['target'][0]
# 
# @tool
# def get_historical_prices(days: int = 10) -> str:
#     """Get closing price sequence for the last 'days' days (for Agent to reference trends)"""
#     if len(df) < days:
#         return str(df['target'].apply(lambda x: x[0]).tolist())
#     prices = df['target'].apply(lambda x: x[0]).tail(days).tolist()
#     return str(prices)
# ========== END OLD TOOLS CODE ==========

tools = [get_current_price, get_historical_prices, get_news]

# -------------------------- 3.5 Technical Indicators --------------------------
def technical_agent(data: pd.DataFrame, current_date: str) -> dict:
    """
    Compute technical indicators and signals up to the current date using full history.

    Returns:
        signal_results: dict with indicator values and signals.
    """
    signal_results = {}

    # Use all historical data up to current_date (avoid look-ahead)
    data_sorted = data.sort_values('date')
    current_mask = data_sorted['start'] <= current_date
    prices = data_sorted.loc[current_mask, 'prices'].astype(float)
    if prices.empty:
        return {
            "signal": "insufficient_data",
            "reason": "No historical price data available up to current date."
        }
    current_price = float(prices.iloc[-1])

    # 1) Trend Signal: EMA(8/21/55)
    if len(prices) >= 55:
        ema_short = prices.ewm(span=8, adjust=False).mean().iloc[-1]
        ema_mid = prices.ewm(span=21, adjust=False).mean().iloc[-1]
        ema_long = prices.ewm(span=55, adjust=False).mean().iloc[-1]
        trend_signal = "bullish" if (ema_short > ema_mid and ema_mid > ema_long) else "bearish"
        signal_results["trend_signal"] = {
            "ema_short_8": round(float(ema_short), 6),
            "ema_mid_21": round(float(ema_mid), 6),
            "ema_long_55": round(float(ema_long), 6),
            "signal": trend_signal
        }
    else:
        signal_results["trend_signal"] = {
            "signal": "insufficient_data",
            "reason": "Need at least 55 data points for EMA(55)."
        }

    # 2) Mean Reversion Signal: Bollinger Bands + Z-score(50) + band position
    if len(prices) >= 50:
        sma20 = prices.rolling(window=20, min_periods=20).mean().iloc[-1]
        std20 = prices.rolling(window=20, min_periods=20).std().iloc[-1]
        upper = sma20 + 2 * std20 if pd.notna(sma20) and pd.notna(std20) else None
        lower = sma20 - 2 * std20 if pd.notna(sma20) and pd.notna(std20) else None

        mean50 = prices.rolling(window=50, min_periods=50).mean().iloc[-1]
        std50 = prices.rolling(window=50, min_periods=50).std().iloc[-1]
        zscore = None
        if pd.notna(mean50) and pd.notna(std50) and std50 != 0:
            zscore = (current_price - mean50) / std50

        band_position = None
        if upper is not None and lower is not None and upper != lower:
            band_position = (current_price - lower) / (upper - lower)

        mean_reversion_signal = "neutral"
        if zscore is not None and band_position is not None:
            if zscore < -2 and band_position <= 0.1:
                mean_reversion_signal = "bullish"
            elif zscore > 2 and band_position >= 0.9:
                mean_reversion_signal = "bearish"

        signal_results["mean_reversion_signal"] = {
            "sma_20": round(float(sma20), 6) if pd.notna(sma20) else None,
            "std_20": round(float(std20), 6) if pd.notna(std20) else None,
            "bollinger_upper": round(float(upper), 6) if upper is not None else None,
            "bollinger_lower": round(float(lower), 6) if lower is not None else None,
            "zscore_50": round(float(zscore), 6) if zscore is not None else None,
            "band_position": round(float(band_position), 6) if band_position is not None else None,
            "signal": mean_reversion_signal
        }
    else:
        signal_results["mean_reversion_signal"] = {
            "signal": "insufficient_data",
            "reason": "Need at least 50 data points for Z-score(50) and Bollinger Bands."
        }

    # 3) RSI Signal: RSI(14)
    if len(prices) >= 14:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean().iloc[-1]
        avg_loss = loss.rolling(window=14, min_periods=14).mean().iloc[-1]

        if pd.isna(avg_gain) or pd.isna(avg_loss):
            rsi_value = None
        elif avg_loss == 0 and avg_gain == 0:
            rsi_value = 50.0
        elif avg_loss == 0:
            rsi_value = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_value = 100 - (100 / (1 + rs))

        rsi_signal = "neutral"
        if rsi_value is not None:
            if rsi_value > 70:
                rsi_signal = "bearish"
            elif rsi_value < 30:
                rsi_signal = "bullish"

        signal_results["rsi_signal"] = {
            "rsi_14": round(float(rsi_value), 6) if rsi_value is not None else None,
            "signal": rsi_signal
        }
    else:
        signal_results["rsi_signal"] = {
            "signal": "insufficient_data",
            "reason": "Need at least 14 data points for RSI(14)."
        }

    # 4) Volatility Signal: 21-day volatility, 63-day mean and z-score
    returns = prices.pct_change()
    if len(returns) >= 63:
        vol_21 = returns.rolling(window=21, min_periods=21).std() * (252 ** 0.5)
        current_vol = vol_21.iloc[-1]
        vol_mean_63 = vol_21.rolling(window=63, min_periods=63).mean().iloc[-1]
        vol_std_63 = vol_21.rolling(window=63, min_periods=63).std().iloc[-1]

        vol_state = None
        vol_zscore = None
        if pd.notna(current_vol) and pd.notna(vol_mean_63) and vol_mean_63 != 0:
            vol_state = current_vol / vol_mean_63
        if pd.notna(current_vol) and pd.notna(vol_mean_63) and pd.notna(vol_std_63) and vol_std_63 != 0:
            vol_zscore = (current_vol - vol_mean_63) / vol_std_63

        vol_signal = "neutral"
        if vol_state is not None and vol_zscore is not None:
            if current_vol < vol_mean_63 and vol_zscore < -1:
                vol_signal = "bullish"
            elif current_vol > vol_mean_63 and vol_zscore > 1:
                vol_signal = "bearish"

        signal_results["volatility_signal"] = {
            "vol_21": round(float(current_vol), 6) if pd.notna(current_vol) else None,
            "vol_mean_63": round(float(vol_mean_63), 6) if pd.notna(vol_mean_63) else None,
            "vol_zscore": round(float(vol_zscore), 6) if vol_zscore is not None else None,
            "vol_state": round(float(vol_state), 6) if vol_state is not None else None,
            "signal": vol_signal
        }
    else:
        signal_results["volatility_signal"] = {
            "signal": "insufficient_data",
            "reason": "Need at least 63 data points for volatility state and z-score."
        }

    # 5) Volume Analysis - skip if no volume data
    volume_columns = [col for col in data.columns if col.lower() in ["volume", "vol", "trade_volume", "trading_volume"]]
    if volume_columns:
        signal_results["volume_analysis"] = {
            "signal": "skipped",
            "reason": "Volume analysis logic not implemented for multiple volume columns."
        }
    else:
        signal_results["volume_analysis"] = {
            "signal": "skipped",
            "reason": "Dataset has no volume column, cannot compute volume-based indicators."
        }

    # 6) Support/Resistance: local min/max in 5-day window
    if len(prices) >= 5:
        supports = []
        resistances = []
        price_list = prices.tolist()
        for idx in range(2, len(price_list) - 2):
            center = price_list[idx]
            left = price_list[idx - 2:idx]
            right = price_list[idx + 1:idx + 3]
            if all(p > center for p in left + right):
                supports.append((idx, center))
            if all(p < center for p in left + right):
                resistances.append((idx, center))

        recent_support = next((price for idx, price in reversed(supports) if idx <= len(price_list) - 3), None)
        recent_resistance = next((price for idx, price in reversed(resistances) if idx <= len(price_list) - 3), None)

        pct_to_support = None
        pct_to_resistance = None
        if recent_support is not None:
            pct_to_support = ((current_price - recent_support) / current_price) * 100
        if recent_resistance is not None:
            pct_to_resistance = ((recent_resistance - current_price) / current_price) * 100

        signal_results["support_resistance"] = {
            "recent_support": round(float(recent_support), 6) if recent_support is not None else None,
            "recent_resistance": round(float(recent_resistance), 6) if recent_resistance is not None else None,
            "pct_to_support": round(float(pct_to_support), 4) if pct_to_support is not None else None,
            "pct_to_resistance": round(float(pct_to_resistance), 4) if pct_to_resistance is not None else None
        }
    else:
        signal_results["support_resistance"] = {
            "signal": "insufficient_data",
            "reason": "Need at least 5 data points for support/resistance detection."
        }

    return signal_results

# -------------------------- 4. LLM & Agent --------------------------
print("[Step 2] Initializing LLM...")
llm = get_llm()
print("[Step 2] LLM initialized successfully.")

print("[Step 3] Creating agent...")
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

# End date configuration (from config.json or None to use termination conditions)
end_date_str = CONFIG.get("end_date")  # Format: "YYYY-MM-DD" or None to use termination conditions
end_date = None
if end_date_str:
    try:
        end_date = pd.to_datetime(end_date_str)
        print(f"[Config] End date specified: {end_date_str}")
    except Exception as e:
        print(f"[Warning] Invalid end_date format: {end_date_str}. Will use termination conditions instead.")
        end_date = None
else:
    print(f"[Config] No end_date specified, will use termination conditions to stop")

# Preserve full history for technical indicators
full_df = df.copy()

# Adjust dataframe to start from specified date
df = df.iloc[start_idx:].reset_index(drop=True)
max_days = len(df) - 1          # Maximum trading days

# Get start date for filename (first date in the adjusted dataframe)
start_date_for_filename = df.iloc[0]['start'] if len(df) > 0 else "unknown"
# Sanitize model name for filename (replace special characters)
model_name_sanitized = LLM_MODEL.replace("/", "-").replace("\\", "-").replace(":", "-")
# Format initial cash for filename (remove decimal point, e.g., 10000.0 -> 10000)
initial_cash_for_filename = str(int(initial_cash)) if initial_cash == int(initial_cash) else str(initial_cash).replace(".", "_")

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

# Create result folder if it doesn't exist
result_dir = Path("result")
result_dir.mkdir(exist_ok=True)

# Generate CSV file paths with start date, model name, and initial cash
main_csv_filename = f"tsla_{start_date_for_filename}_{model_name_sanitized}_{initial_cash_for_filename}.csv"
llm_outputs_csv_filename = f"tsla_llm_outputs_{start_date_for_filename}_{model_name_sanitized}_{initial_cash_for_filename}.csv"
llm_inputs_csv_filename = f"tsla_llm_inputs_{start_date_for_filename}_{model_name_sanitized}_{initial_cash_for_filename}.csv"
main_csv_path = result_dir / main_csv_filename
llm_outputs_csv_path = result_dir / llm_outputs_csv_filename
llm_inputs_csv_path = result_dir / llm_inputs_csv_filename

print(f"[Step 4] Results will be saved to:")
print(f"  - Main results: {main_csv_path}")
print(f"  - LLM outputs: {llm_outputs_csv_path}")
print(f"  - LLM inputs: {llm_inputs_csv_path}")

# Initialize CSV files with headers (if files don't exist)
# We'll create empty DataFrames with the expected columns to get the headers
main_columns = [
    "date", "hold_shares", "decision", "requested_quantity", "actual_quantity",
    "market_value", "daily_gross_pnl", "trade_count",
    "daily_commission_cost", "daily_trading_cost",
    "daily_llm_input_tokens", "daily_llm_output_tokens", "daily_llm_cost_usd",
    "daily_infra_cost", "daily_random_cost", "monthly_data_subscription_cost",
    "daily_total_cost", "cumulative_cost", "cumulative_net_profit", "slippage_usd"
]
llm_columns = [
    "date", "day", "current_price", "available_cash", "current_shares",
    "decision", "requested_quantity", "actual_quantity", "llm_output",
    "input_tokens", "output_tokens", "llm_cost_usd"
]
llm_inputs_columns = [
    "date", "input_content", "input_tokens", "output_tokens", "total_tokens"
]

# Create CSV files with headers if they don't exist
if not main_csv_path.exists():
    pd.DataFrame(columns=main_columns).to_csv(main_csv_path, index=False)
if not llm_outputs_csv_path.exists():
    pd.DataFrame(columns=llm_columns).to_csv(llm_outputs_csv_path, index=False)
if not llm_inputs_csv_path.exists():
    pd.DataFrame(columns=llm_inputs_columns).to_csv(llm_inputs_csv_path, index=False)

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
    current_price = float(current_row['prices'])  # Day t closing price → visible to Agent
    
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
    history_prices = df.iloc[max(0, i-9):i+1]['prices'].tolist()
    
    # Technical indicators and signals
    signal_results = technical_agent(full_df, date_str)

    # Get news for current date (directly include in prompt instead of requiring tool call)
    try:
        if 'news' not in df.columns:
            news_content = "News column not available in dataset."
        else:
            current_news = df.iloc[i]['news']
            
            # Handle different news formats
            if pd.isna(current_news) or current_news is None:
                news_content = "No news available for this date."
            elif isinstance(current_news, list):
                if len(current_news) > 0:
                    # Check if list contains strings or other types
                    if isinstance(current_news[0], str):
                        # Filter out empty strings
                        valid_articles = [article for article in current_news if article and isinstance(article, str) and article.strip()]
                        if valid_articles:
                            news_content = "\n\n".join([f"News Article {idx+1}:\n{article}" for idx, article in enumerate(valid_articles)])
                        else:
                            news_content = "No news available for this date."
                    else:
                        # List contains non-string items, convert to string
                        news_content = "\n\n".join([f"News Article {idx+1}:\n{str(article)}" for idx, article in enumerate(current_news) if article])
                        if not news_content:
                            news_content = "No news available for this date."
                else:
                    news_content = "No news available for this date."
            elif isinstance(current_news, str):
                if current_news.strip():
                    news_content = current_news
                else:
                    news_content = "No news available for this date."
            else:
                # Try to convert to string
                news_str = str(current_news)
                if news_str and news_str.strip() and news_str.lower() not in ['nan', 'none', 'null']:
                    news_content = news_str
                else:
                    news_content = "No news available for this date."
    except KeyError:
        news_content = "News column not available in dataset."
    except Exception as e:
        news_content = f"Error retrieving news: {str(e)}"
    
    # Get recent news (last 3 days including today) for context
    recent_news_list = []
    if 'news' in df.columns:
        for j in range(max(0, i-2), i+1):
            try:
                day_news = df.iloc[j]['news']
                day_date = df.iloc[j]['start']
                
                if pd.isna(day_news) or day_news is None:
                    continue
                elif isinstance(day_news, list) and len(day_news) > 0:
                    # Filter valid articles
                    valid_articles = [article for article in day_news[:2] if article and isinstance(article, str) and article.strip()]
                    if valid_articles:
                        formatted_news = "\n".join([f"  - {article[:200]}..." if len(article) > 200 else f"  - {article}" for article in valid_articles])
                        recent_news_list.append(f"Date {day_date}:\n{formatted_news}")
                elif isinstance(day_news, str) and day_news.strip():
                    formatted_news = f"  - {day_news[:200]}..." if len(day_news) > 200 else f"  - {day_news}"
                    recent_news_list.append(f"Date {day_date}:\n{formatted_news}")
            except Exception:
                pass
    
    recent_news_summary = "\n\n".join(recent_news_list) if recent_news_list else "No recent news available."

    # Agent decision + record actual token consumption
    try:
        with get_openai_callback() as cb:
            # Use new API messages format
            messages = [
                SystemMessage(content="""
You are a trading agent analyzing market data to make investment decisions about TSLA stock.
Based on your analysis, provide a specific recommendation to buy, sell, or hold. 

IMPORTANT: You will receive the current trading date in each request. Pay attention to the date as it helps you understand the temporal context of the market data.

MANDATORY REQUIREMENT: You MUST analyze the news information provided in each request. News articles contain critical information about market sentiment, company developments, regulatory changes, earnings, product launches, and other factors that significantly impact stock prices. Your decision MUST be based on BOTH price data AND news analysis. Ignoring news information will lead to poor trading decisions.

AVAILABLE TOOLS (optional, news is already provided in the prompt):
You have access to the following tools if you need additional information:
1. get_current_price(date): Get TSLA closing price for a specific date
2. get_historical_prices(days): Get closing price sequence for the last N days
3. get_news(date): Get TSLA news articles for a specific date (use if you need news from other dates)

Do not forget to utilize lessons from past decisions to learn from your mistakes. 
Here are some reflections from similar situations you traded in and the lessons learned:
- Consider market trends and volatility patterns
- Balance risk and reward based on available capital
- Avoid emotional trading decisions
- Learn from previous gains and losses
- ALWAYS analyze news alongside price data - news often provides context for price movements

IMPORTANT: You must provide your decision reasoning BEFORE stating your decision.

Output format:
1. First, explain your analysis and reasoning for the decision (decision rationale)
   - You MUST explain how the news information influenced your decision
   - Analyze both positive and negative news signals
   - Consider how news relates to price trends
2. Then, conclude with your decision in the exact format:
   - For buy: "buy <number>" where <number> is the number of shares to buy (positive integer)
   - For sell: "sell <number>" where <number> is the number of shares to sell (positive integer)
   - For hold: "hold" (no number needed)

Examples:
- "Based on the price trend showing a 10% increase over the last week, positive news about production expansion indicating strong demand, and current momentum, I recommend buying. The news suggests continued growth potential. buy 100"
- "The stock price has been declining for 5 consecutive days with high volatility, and recent news indicates regulatory concerns and potential fines. This combination of negative price action and negative news suggests potential further decline. sell 50"
- "The price is relatively stable with no clear trend, and recent news is neutral with no significant developments. I want to wait for more signals before making a move. hold"

Always provide your reasoning before the decision statement.
"""),
                # ========== OLD PROMPT (COMMENTED OUT) ==========
                # SystemMessage(content="""
                # You are a trading agent analyzing market data to make investment decisions about TSLA stock.
                # Based on your analysis, provide a specific recommendation to buy, sell, or hold. 
                # 
                # IMPORTANT: You will receive the current trading date in each request. Pay attention to the date as it helps you understand the temporal context of the market data.
                # 
                # AVAILABLE TOOLS:
                # You have access to the following tools to help you make informed decisions:
                # 1. get_current_price(date): Get TSLA closing price for a specific date
                # 2. get_historical_prices(days): Get closing price sequence for the last N days
                # 3. get_news(date): Get TSLA news articles for a specific date - USE THIS to access relevant news that may affect stock price
                # 
                # IMPORTANT: You are STRONGLY ENCOURAGED to use the get_news tool to check news for the current trading date and recent dates. News articles contain valuable information about market sentiment, company developments, regulatory changes, and other factors that can significantly impact stock prices. Always consider news alongside price data when making your decisions.
                # 
                # Do not forget to utilize lessons from past decisions to learn from your mistakes. 
                # Here are some reflections from similar situations you traded in and the lessons learned:
                # - Consider market trends and volatility patterns
                # - Balance risk and reward based on available capital
                # - Avoid emotional trading decisions
                # - Learn from previous gains and losses
                # - Consider news and market sentiment when available
                # 
                # IMPORTANT: You must provide your decision reasoning BEFORE stating your decision.
                # 
                # Output format:
                # 1. First, explain your analysis and reasoning for the decision (decision rationale)
                #    - If you used news information, mention how it influenced your decision
                # 2. Then, conclude with your decision in the exact format:
                #    - For buy: "buy <number>" where <number> is the number of shares to buy (positive integer)
                #    - For sell: "sell <number>" where <number> is the number of shares to sell (positive integer)
                #    - For hold: "hold" (no number needed)
                # 
                # Examples:
                # - "Based on the price trend showing a 10% increase over the last week, positive news about production expansion, and current momentum, I recommend buying. buy 100"
                # - "The stock price has been declining for 5 consecutive days with high volatility, and recent news indicates regulatory concerns. This suggests potential further decline. sell 50"
                # - "The price is relatively stable with no clear trend, and recent news is neutral. I want to wait for more signals. hold"
                # 
                # Always provide your reasoning before the decision statement.
                # """),
                # ========== END OLD PROMPT ==========
                HumanMessage(content=f"""Trading Date: {date_str}

=== PRICE DATA ===
Last 10 days closing price sequence: {history_prices}
Today's closing price: ${current_price:.3f}

=== PORTFOLIO STATUS ===
Available cash: ${cash:.2f}
Current shares: {shares}

=== TECHNICAL SIGNALS ===
{json.dumps(signal_results, ensure_ascii=False)}

=== NEWS INFORMATION (MANDATORY TO ANALYZE) ===
Today's News ({date_str}):
{news_content}

Recent News Summary (Last 3 days):
{recent_news_summary}

=== INSTRUCTIONS ===
Please analyze BOTH the price data AND the news information provided above. You MUST consider how the news affects market sentiment and stock price movements. Based on your comprehensive analysis of price trends and news content, provide your investment decision in the format: buy <number>, sell <number>, or hold:""")
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
            
            # Build input content string (SystemMessage + HumanMessage)
            input_content_parts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    input_content_parts.append(f"[SYSTEM MESSAGE]\n{msg.content}")
                elif isinstance(msg, HumanMessage):
                    input_content_parts.append(f"[HUMAN MESSAGE]\n{msg.content}")
            input_content = "\n\n".join(input_content_parts)
            
            # Record LLM input content, tokens, and date
            llm_input_record = {
                "date": date_str,
                "input_content": input_content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            
            # Immediately write to CSV file (append mode)
            pd.DataFrame([llm_input_record]).to_csv(llm_inputs_csv_path, mode='a', header=False, index=False)
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

    # Next day execution price (more accurate: next row prices)
    if i + 1 >= len(df):
        break
    exec_price = float(df.iloc[i + 1]['prices'])

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
    record = {
        "date": date_str,
        "hold_shares": shares,
        "decision": action,
        "requested_quantity": quantity,  # Agent requested quantity
        "actual_quantity": actual_quantity,  # Actually executed quantity
        "market_value": round(market_value, 2),
        "daily_gross_pnl": round(cumulative_gross_profit - (records[-1]['market_value'] + records[-1]['daily_gross_pnl'] if records else initial_cash), 2),
        "trade_count": trade_count,
        "daily_commission_cost": round(commission, 4),
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
    }
    records.append(record)
    
    # Immediately write to CSV file (append mode)
    pd.DataFrame([record]).to_csv(main_csv_path, mode='a', header=False, index=False)
    
    # Record LLM output separately
    llm_record = {
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
    }
    llm_outputs_records.append(llm_record)
    
    # Immediately write to CSV file (append mode)
    pd.DataFrame([llm_record]).to_csv(llm_outputs_csv_path, mode='a', header=False, index=False)

    # Termination judgment
    # If end_date is set, only check end date and ignore other termination conditions
    if end_date is not None:
        if current_date >= end_date:
            print(f"[End Date Reached] Date: {date_str}, Reached specified end date: {end_date_str}, Ran for {i+1} days")
            break
    else:
        # Only check other termination conditions if end_date is not set
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
# Results have been written to CSV files in real-time during the loop
result_df = pd.DataFrame(records)

print("\nBacktest completed! Results saved:")
print(f"  - Main results: {main_csv_path}")
print(f"  - LLM outputs: {llm_outputs_csv_path}")
print(f"  - LLM inputs: {llm_inputs_csv_path}")
print(f"Total trading days: {len(result_df)}")
if len(result_df) > 0:
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