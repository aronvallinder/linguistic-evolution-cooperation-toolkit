import yaml
from itertools import product
from typing import Dict, List

class ExperimentConfig:
    """
    Loads `config/experiments.yaml` and expands a named experiment set into a list
    of concrete run configurations.

    Key idea:
    - An experiment set defines lists (or the string "all") for:
      models × templates × personas × task_orders
    - We expand those into a cartesian product so the batch runner can iterate
      and run each combination deterministically.
    """
    def __init__(self, config_path: str):
        # Read YAML once at initialization; subsequent calls reuse `self.config`.
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_experiment_combinations(self, experiment_name: str) -> List[Dict]:
        """
        Generate all parameter combinations for an experiment set.

        Returns a list of dicts (one per run), where each dict has:
        - model: resolved provider model id string (from base_models)
        - template: resolved system prompt template string (from prompt_templates)
        - persona: resolved persona object (from personas)
        - task_order: list like ["game","myth"] (from game_parameters.task_orders)
        - game_params: default game_parameters merged with any per-set overrides
        """
        exp_set = self.config['experiment_sets'][experiment_name]

        # Resolve "all" references
        # If a field is set to "all" in YAML, we expand it to all keys in that section
        # (or, for task_orders, all values under game_parameters.task_orders).
        models = self._resolve_all(exp_set['models'], 'base_models')
        templates = self._resolve_all(exp_set['templates'], 'prompt_templates')
        personas = self._resolve_all(exp_set['personas'], 'personas')
        task_orders = self._resolve_all(exp_set['task_orders'], 'task_orders',
                                       from_game_params=True)

        # Generate all combinations
        # Cartesian product across the expanded dimensions.
        # Example: 3 models × 2 personas × 4 task_orders × 1 template = 24 runs.
        combinations = []
        for model, template, persona, order in product(models, templates, personas, task_orders):
            combo = {
                # Resolve symbolic names (like "llama8b") into actual provider ids/objects.
                'model': self.config['base_models'][model],
                'template': self.config['prompt_templates'][template],
                'persona': self.config['personas'][persona],
                'task_order': order,
                # Merge default game parameters with any per-experiment overrides.
                'game_params': self._get_game_params(exp_set.get('game_params', {}))
            }
            combinations.append(combo)

        return combinations

    def _resolve_all(self, param, config_key, from_game_params=False):
        # Helper: expand "all" to either:
        # - all keys in the referenced config section (e.g., base_models / personas), OR
        # - for task_orders: all task order lists in game_parameters.task_orders.
        if param == "all":
            if from_game_params:
                return self.config['game_parameters'][config_key]
            return list(self.config[config_key].keys())
        return param

    def _get_game_params(self, exp_game_params: Dict) -> Dict:
        # Merge experiment-specific game params with defaults.
        # This lets you override e.g. num_turns/temperature per experiment set.
        default_params = self.config['game_parameters'].copy()
        default_params.update(exp_game_params)
        return default_params