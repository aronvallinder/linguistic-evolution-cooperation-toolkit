# Modular LLM Experimental Framework

Configuration-driven experiments for multi-agent LLM behavior in a repeated Trust (Investment) Game, optionally paired with a myth-writing task. Runs save a canonical **simulation state JSON**, and analysis scripts turn those JSONs into standardized plots/metrics.

## Quick start

- **Install**:
  - `pip install -r requirements.txt`
- **Set API keys**:
  - Copy `env.example` → `.env` and export variables in your shell:
    - `OPENROUTER_API_KEY` (for simulations via OpenRouter)
    - `TOGETHER_API_KEY` (only for LLM-judge myth similarity)

## Run experiments

- **Single run**: `python experiments/run_trust_game.py`
- **Batch from config**: `python experiments/run_trust_game_batch.py <experiment_set>`

## Run analyses

Use the unified runner:

- `./scripts/run_all_analyses.sh -all <input_json> <output_dir> [task_name]`

## Documentation

- See the fuller project guide in `docs/README.md`.


