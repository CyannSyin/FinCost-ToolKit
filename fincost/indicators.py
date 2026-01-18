from __future__ import annotations

import pandas as pd


def technical_agent(data: pd.DataFrame, current_date: str) -> dict:
    """
    Compute technical indicators and signals up to the current date using full history.
    """
    signal_results = {}

    data_sorted = data.sort_values("date")
    current_mask = data_sorted["start"] <= current_date
    prices = data_sorted.loc[current_mask, "prices"].astype(float)
    if prices.empty:
        return {
            "signal": "insufficient_data",
            "reason": "No historical price data available up to current date.",
        }
    current_price = float(prices.iloc[-1])

    if len(prices) >= 55:
        ema_short = prices.ewm(span=8, adjust=False).mean().iloc[-1]
        ema_mid = prices.ewm(span=21, adjust=False).mean().iloc[-1]
        ema_long = prices.ewm(span=55, adjust=False).mean().iloc[-1]
        trend_signal = (
            "bullish" if (ema_short > ema_mid and ema_mid > ema_long) else "bearish"
        )
        signal_results["trend_signal"] = {
            "ema_short_8": round(float(ema_short), 6),
            "ema_mid_21": round(float(ema_mid), 6),
            "ema_long_55": round(float(ema_long), 6),
            "signal": trend_signal,
        }
    else:
        signal_results["trend_signal"] = {
            "signal": "insufficient_data",
            "reason": "Need at least 55 data points for EMA(55).",
        }

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
            "band_position": round(float(band_position), 6)
            if band_position is not None
            else None,
            "signal": mean_reversion_signal,
        }
    else:
        signal_results["mean_reversion_signal"] = {
            "signal": "insufficient_data",
            "reason": "Need at least 50 data points for Z-score(50) and Bollinger Bands.",
        }

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
            "signal": rsi_signal,
        }
    else:
        signal_results["rsi_signal"] = {
            "signal": "insufficient_data",
            "reason": "Need at least 14 data points for RSI(14).",
        }

    returns = prices.pct_change()
    if len(returns) >= 63:
        vol_21 = returns.rolling(window=21, min_periods=21).std() * (252**0.5)
        current_vol = vol_21.iloc[-1]
        vol_mean_63 = vol_21.rolling(window=63, min_periods=63).mean().iloc[-1]
        vol_std_63 = vol_21.rolling(window=63, min_periods=63).std().iloc[-1]

        vol_state = None
        vol_zscore = None
        if pd.notna(current_vol) and pd.notna(vol_mean_63) and vol_mean_63 != 0:
            vol_state = current_vol / vol_mean_63
        if (
            pd.notna(current_vol)
            and pd.notna(vol_mean_63)
            and pd.notna(vol_std_63)
            and vol_std_63 != 0
        ):
            vol_zscore = (current_vol - vol_mean_63) / vol_std_63

        vol_signal = "neutral"
        if vol_state is not None and vol_zscore is not None:
            if current_vol < vol_mean_63 and vol_zscore < -1:
                vol_signal = "bullish"
            elif current_vol > vol_mean_63 and vol_zscore > 1:
                vol_signal = "bearish"

        signal_results["volatility_signal"] = {
            "vol_21": round(float(current_vol), 6) if pd.notna(current_vol) else None,
            "vol_mean_63": round(float(vol_mean_63), 6)
            if pd.notna(vol_mean_63)
            else None,
            "vol_zscore": round(float(vol_zscore), 6) if vol_zscore is not None else None,
            "vol_state": round(float(vol_state), 6) if vol_state is not None else None,
            "signal": vol_signal,
        }
    else:
        signal_results["volatility_signal"] = {
            "signal": "insufficient_data",
            "reason": "Need at least 63 data points for volatility state and z-score.",
        }

    volume_columns = [
        col
        for col in data.columns
        if col.lower() in ["volume", "vol", "trade_volume", "trading_volume"]
    ]
    if volume_columns:
        signal_results["volume_analysis"] = {
            "signal": "skipped",
            "reason": "Volume analysis logic not implemented for multiple volume columns.",
        }
    else:
        signal_results["volume_analysis"] = {
            "signal": "skipped",
            "reason": "Dataset has no volume column, cannot compute volume-based indicators.",
        }

    if len(prices) >= 5:
        supports = []
        resistances = []
        price_list = prices.tolist()
        for idx in range(2, len(price_list) - 2):
            center = price_list[idx]
            left = price_list[idx - 2 : idx]
            right = price_list[idx + 1 : idx + 3]
            if all(p > center for p in left + right):
                supports.append((idx, center))
            if all(p < center for p in left + right):
                resistances.append((idx, center))

        recent_support = next(
            (price for idx, price in reversed(supports) if idx <= len(price_list) - 3),
            None,
        )
        recent_resistance = next(
            (
                price
                for idx, price in reversed(resistances)
                if idx <= len(price_list) - 3
            ),
            None,
        )

        pct_to_support = None
        pct_to_resistance = None
        if recent_support is not None:
            pct_to_support = ((current_price - recent_support) / current_price) * 100
        if recent_resistance is not None:
            pct_to_resistance = ((recent_resistance - current_price) / current_price) * 100

        signal_results["support_resistance"] = {
            "recent_support": round(float(recent_support), 6)
            if recent_support is not None
            else None,
            "recent_resistance": round(float(recent_resistance), 6)
            if recent_resistance is not None
            else None,
            "pct_to_support": round(float(pct_to_support), 4)
            if pct_to_support is not None
            else None,
            "pct_to_resistance": round(float(pct_to_resistance), 4)
            if pct_to_resistance is not None
            else None,
        }
    else:
        signal_results["support_resistance"] = {
            "signal": "insufficient_data",
            "reason": "Need at least 5 data points for support/resistance detection.",
        }

    return signal_results
