from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def load_config() -> Dict[str, Any]:
    """Load config.json if it exists, otherwise return empty dict."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"Failed to read config.json, will use default configuration: {exc}")
            return {}
    return {}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None

    start_date: str | None = None
    end_date: str | None = None
    initial_cash: float = 10000.0

    commission_type: str = "fixed"
    commission_per_share: float = 0.005
    commission_minimum: float = 1.0
    commission_maximum_rate: float = 0.01
    infra_daily: float = 0.5
    random_cost_min: float = 0.0
    random_cost_max: float = 1.0

    input_price_per_k_tokens: float = 0.00015 / 1000
    output_price_per_k_tokens: float = 0.00060 / 1000

    data_subscription_monthly: float = 0.0

    @property
    def random_cost_range(self) -> Tuple[float, float]:
        return (self.random_cost_min, self.random_cost_max)


def settings_from_config(config: Dict[str, Any]) -> Settings:
    return Settings(
        llm_provider=str(config.get("llm_provider", "openai")).lower(),
        llm_model=str(config.get("llm_model", "gpt-4o-mini")),
        llm_base_url=config.get("llm_base_url"),
        start_date=config.get("start_date"),
        end_date=config.get("end_date"),
        initial_cash=float(config.get("initial_cash", 10000.0)),
        commission_type=config.get("commission_type", "fixed"),
        commission_per_share=float(config.get("commission_per_share", 0.005)),
        commission_minimum=float(config.get("commission_minimum", 1.0)),
        commission_maximum_rate=float(config.get("commission_maximum_rate", 0.01)),
        infra_daily=float(config.get("infra_daily", 0.5)),
        random_cost_min=float(config.get("random_cost_min", 0.0)),
        random_cost_max=float(config.get("random_cost_max", 1.0)),
        input_price_per_k_tokens=float(config.get("input_price_per_k_tokens", 0.00015 / 1000)),
        output_price_per_k_tokens=float(config.get("output_price_per_k_tokens", 0.00060 / 1000)),
        data_subscription_monthly=float(config.get("data_subscription_monthly", 0.0)),
    )


def load_settings() -> Settings:
    return settings_from_config(load_config())
