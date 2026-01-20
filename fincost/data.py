from __future__ import annotations

from typing import Tuple

import pandas as pd
from datasets import load_dataset

from .config import Settings


def load_price_data() -> pd.DataFrame:
    print("[Step 1] Loading dataset from Hugging Face...")
    dataset = load_dataset("TheFinAI/CLEF_Task3_Trading", split="TSLA")
    print("[Step 1] Dataset loaded successfully.")
    print(f"[Step 1] Loaded TSLA split with {len(dataset)} rows")

    print("[Step 1] Converting to DataFrame...")
    df = dataset.to_pandas()
    print(f"[Step 1] DataFrame columns: {list(df.columns)}")
    print(f"[Step 1] DataFrame shape: {df.shape}")

    required_columns = ["date", "prices"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(
            f"[ERROR] Required columns not found: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )
        raise KeyError(f"Required columns not found: {missing_columns}")

    if "news" not in df.columns:
        print(
            f"[Warning] 'news' column not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
        print("[Warning] News information will not be available during backtest.")
    else:
        news_sample_count = df["news"].notna().sum()
        print(f"[Step 1] News column found: {news_sample_count}/{len(df)} rows have news data")
        if news_sample_count > 0:
            sample_news = df[df["news"].notna()].iloc[0]["news"]
            print(f"[Step 1] Sample news type: {type(sample_news)}")
            if isinstance(sample_news, list):
                print(f"[Step 1] Sample news is list with {len(sample_news)} items")
            elif isinstance(sample_news, str):
                print(f"[Step 1] Sample news is string with length {len(sample_news)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["start"] = df["date"].dt.strftime("%Y-%m-%d")

    print(
        f"[Step 1] Data loaded: {len(df)} rows, "
        f"Date range: {df['date'].min().date()} ~ {df['date'].max().date()}"
    )
    return df


def apply_config_dates(
    df: pd.DataFrame, settings: Settings
) -> Tuple[pd.DataFrame, str, pd.Timestamp | None]:
    start_date_str = settings.start_date
    if start_date_str:
        start_date_idx = df[df["start"] == start_date_str].index
        if len(start_date_idx) > 0:
            start_idx = start_date_idx[0]
            print(f"[Config] Starting from date: {start_date_str} (index: {start_idx})")
        else:
            print(
                f"[Warning] Start date {start_date_str} not found in dataset. "
                "Starting from beginning."
            )
            start_idx = 0
    else:
        start_idx = 0
        print(
            "[Config] No start_date specified, starting from first date in dataset: "
            f"{df.iloc[0]['start']}"
        )

    end_date = None
    end_date_str = settings.end_date
    if end_date_str:
        try:
            end_date = pd.to_datetime(end_date_str)
            print(f"[Config] End date specified: {end_date_str}")
        except Exception:
            print(
                f"[Warning] Invalid end_date format: {end_date_str}. "
                "Will use termination conditions instead."
            )
            end_date = None
    else:
        print("[Config] No end_date specified, will use termination conditions to stop")

    df = df.iloc[start_idx:].reset_index(drop=True)
    start_date_for_filename = df.iloc[0]["start"] if len(df) > 0 else "unknown"
    return df, start_date_for_filename, end_date


def build_data(
    settings: Settings,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, pd.Timestamp | None]:
    df = load_price_data()
    full_df = df.copy()
    df, start_date_for_filename, end_date = apply_config_dates(df, settings)
    return df, full_df, start_date_for_filename, end_date
