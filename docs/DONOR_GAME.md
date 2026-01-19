# Donor Game

The Donor Game is a population-based economic game that models social dilemmas involving resource sharing and cooperation. It is implemented as part of the Linguistic Evolution Cooperation Toolkit.

## Overview

In the Donor Game, a population of agents are randomly paired each round. Each agent decides how much of their resources to donate to their partner. Donations are multiplied by a configurable factor before reaching the recipient, creating a positive-sum environment where mutual cooperation benefits everyone.

## Game Rules

### Setup
- A population of **N agents** (must be even for pairing)
- Each agent starts with an **initial resource amount** (e.g., 100 units)
- A **donation multiplier** determines how much recipients gain (e.g., 2x means donating 10 units gives the partner 20)

### Each Round
1. **Random Pairing**: Agents are randomly paired into bidirectional partnerships
2. **Simultaneous Decisions**: Both agents in a pair independently decide how much to donate
3. **Resource Transfer**: Donations are processed simultaneously
   - The donor loses the donated amount
   - The recipient gains the donated amount × multiplier

### Constraints
- Agents can donate any amount from **0 to their current balance**
- Agents **cannot go into negative resources**
- Donations are **irreversible** within a round

## Decision Process

Each round, agents follow a two-step decision process:

1. **Reasoning Phase**: The agent receives information about the current situation and their partner's observable history, then provides reasoning about their donation decision
2. **Donation Phase**: The agent specifies the exact amount to donate

## Observable Information

Agents have access to:
- Their **current resource balance**
- Their **partner's identity** for this round
- **Observable history** of their partner's past donations (configurable depth and width)
- The **donation multiplier**

The history tree system allows agents to see:
- What their partner donated in previous rounds
- What those recipients subsequently donated (recursive observation)

History visibility is controlled by:
- **context_depth**: How many levels of donation chains to show
- **context_width**: How many donations to show per level

## Configuration Parameters

### Game Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `initial_resources` | Starting resources per agent | 100 |
| `donation_multiplier` | Multiplication factor for donations | 2.0 |
| `population_size` | Number of agents (must be even) | 10 |
| `context_depth` | Depth of observable history tree | 1 |
| `context_width` | Width of observable history tree | 1 |
| `personas` | Optional personality traits per agent | See below |
| `system_prompt_template` | Custom system prompt for agents | See below |

### Simulation Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model` | LLM model identifier | `"gpt-4o"` |
| `temperature` | LLM sampling temperature | 0.8 |
| `num_turns` | Number of rounds to play | 10 |
| `memory_capacity` | How many past messages agents retain | 5 |
| `agent_biases` | Optional list of initial biases per agent | `None` |
| `task_order` | Tasks to execute each round | `["game", "myth"]` |
| `myth_visibility` | Which agents' myths each agent can see | `"all"` |
| `myth_partner_rounds` | How many rounds back to look for partners (for `recent_partners` mode) | 1 |

## Task Order

The `task_order` parameter controls what activities occur each round:

| Task Order | Description |
|------------|-------------|
| `["game"]` | Only play the donation game (no myth writing) |
| `["game", "myth"]` | Play the game first, then write myths |
| `["myth", "game"]` | Write myths first, then play the game |
| `["myth"]` | Only write myths (no game play) |

When myth writing is enabled, you must also provide a `myth_writer` with a topic:

```python
myth_writer = MythWriter(myth_topic="cooperation and sharing")
```

## Myth Visibility

When myth writing is enabled (by including `"myth"` in the `task_order`), agents write short myths each round. The `myth_visibility` parameter controls which other agents' myths each agent can see when writing their own:

| Mode | Description |
|------|-------------|
| `"all"` | Each agent sees myths from **all other agents** from the previous round. This creates maximum information sharing and allows myths to spread and evolve across the entire population. |
| `"recent_partners"` | Each agent only sees myths from agents they have been **paired with recently**. The `myth_partner_rounds` parameter controls how many rounds back to look for partners. This creates more localized myth evolution, where stories spread through direct interaction networks. |

### Example Configuration

```yaml
# All agents see all myths (default)
myth_visibility: "all"

# Agents only see myths from partners in the last 2 rounds
myth_visibility: "recent_partners"
myth_partner_rounds: 2
```

### Implications for Myth Evolution

- **`"all"` mode**: Myths converge more quickly as all agents are exposed to the same set of stories. Good for studying population-wide narrative evolution.
- **`"recent_partners"` mode**: Myths evolve in localized clusters based on interaction patterns. Good for studying how information spreads through social networks and whether distinct narrative traditions emerge in different parts of the population.

## Agent Personas

Agents can be assigned personas that influence their behavior through system prompt additions:

```yaml
personas:
  Agent_1:
    system_addition: "You believe in generous cooperation and building trust."
  Agent_2:
    system_addition: "You are cautious and prefer to verify trustworthiness before giving."
```

Pre-defined personas are available in `config/experiments.yaml`:
- `neutral` - No additional behavioral guidance
- `altruistic` - Prioritizes others' welfare
- `selfish` - Prioritizes own interests
- `cautious` - Risk-averse, conservative choices
- `risk_taking` - Comfortable with bold moves

## Custom System Prompts

You can provide a custom `system_prompt_template` to override the default agent instructions. The template can use placeholders that get filled with game parameters:

```python
custom_template = """You are participating in a Donor Game with {population_size} agents.

GAME RULES:
- Each agent starts with {initial_resources} resource units
- Donations are multiplied by {donation_multiplier}x
...
"""

game = DonorGame(
    initial_resources=100,
    donation_multiplier=2,
    population_size=6,
    system_prompt_template=custom_template,
)
```

## LLM Configuration

The simulation auto-detects your LLM provider based on environment variables:

| Environment Variable | Provider |
|---------------------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `GEMINI_API_KEY` | Google Gemini |
| `OPENROUTER_API_KEY` | OpenRouter |

You can force a specific provider by setting `LLM_PROVIDER`:

```bash
export LLM_PROVIDER="openai"  # or "gemini", "openrouter"
```

Example models by provider:
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **Gemini**: `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`
- **OpenRouter**: `meta-llama/llama-3.1-8b-instruct`, `meta-llama/llama-3.3-70b-instruct`

## Output and Analysis

The game tracks:
- **Per-round donations**: Who donated how much to whom
- **Donation percentages**: Proportion of resources donated
- **Balance trajectories**: How each agent's resources change over time
- **Pairing history**: Which agents were paired in each round
- **Myths** (if enabled): Each agent's myth per round

Results are saved to JSON using `sim_data.save_state(filepath)`:

```python
sim_data.save_state("data/json/donor_game/my_experiment.json")
```

The JSON output includes:
- `conversation_history`: Round-by-round record of pairings, donations, balances, and myths
- `game_data`: Cumulative donation history and balance tracking
- `agents`: Per-agent configuration (model, memory capacity, system prompt, message history)

## Example Scenario

**Setup**: 4 agents, 100 initial resources, 2x multiplier

**Round 1 Pairings**: (Agent_1, Agent_2), (Agent_3, Agent_4)

| Donor | Recipient | Amount | Recipient Receives |
|-------|-----------|--------|-------------------|
| Agent_1 | Agent_2 | 50 | 100 |
| Agent_2 | Agent_1 | 30 | 60 |
| Agent_3 | Agent_4 | 0 | 0 |
| Agent_4 | Agent_3 | 40 | 80 |

**Balances after Round 1**:
- Agent_1: 100 - 50 + 60 = 110
- Agent_2: 100 - 30 + 100 = 170
- Agent_3: 100 - 0 + 80 = 180
- Agent_4: 100 - 40 + 0 = 60


