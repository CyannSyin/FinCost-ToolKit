<<<<<<< HEAD
# FinCost-Demo
=======
# FinCost-Demo

TSLA backtest demo with LLM-driven trading decisions.

## Quick Start (uv)

1. Create and activate a virtual environment:
   - `uv venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `uv pip install -e .`
3. Configure environment variables:
   - `cp env.example .env`
   - Fill in API keys in `.env`
4. Adjust config (optional):
   - Edit `config.json` for model/provider, dates, and costs
5. Run:
   - `python main.py`

## Notes

- The legacy monolithic script remains in `demo.py`.
- Outputs are saved to `result/` as CSV files.
- Visualization still runs from `visualization.py` (point it to a CSV file).
>>>>>>> tiger
