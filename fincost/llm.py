from __future__ import annotations

import os
import sys

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .config import Settings


def get_llm(settings: Settings):
    if settings.llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Missing OPENAI_API_KEY")
            sys.exit(1)
        try:
            llm = ChatOpenAI(model=settings.llm_model, temperature=0)
            print(f"Using OpenAI model: {settings.llm_model}")
            return llm
        except Exception as exc:
            print(f"Failed to initialize OpenAI LLM: {exc}")
            raise
    if settings.llm_provider == "aihubmix":
        api_key = os.getenv("AIHUBMIX_API_KEY")
        if not api_key:
            print("Missing AIHUBMIX_API_KEY")
            sys.exit(1)
        base_url = settings.llm_base_url
        if not base_url:
            print("Missing AIHUBMIX_BASE_URL (required in config.json or .env)")
            sys.exit(1)
        try:
            llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=0,
                base_url=base_url,
                api_key=api_key,
            )
            print(f"Using Aihubmix model: {settings.llm_model} at {base_url}")
            return llm
        except Exception as exc:
            print(f"Failed to initialize Aihubmix LLM: {exc}")
            raise
    raise NotImplementedError(f"LLM_PROVIDER not implemented: {settings.llm_provider}")


def create_trading_agent(llm, tools):
    return create_agent(llm, tools=tools)
