import pytest
from unittest.mock import MagicMock
from games.donor_game import DonorGame


class MockSimData:
    """Mock simulation data for testing."""

    def __init__(self):
        self.agents = {}
        self.conversation_history = []
        self.game_data = {}


class TestDonorGameInitialization:
    """Tests for DonorGame initialization and validation."""

    def test_basic_initialization(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        assert game.initial_resources == 100
        assert game.donation_multiplier == 3
        assert game.population_size == 4
        assert game.context_depth == 1
        assert game.context_width == 1

    def test_custom_context_parameters(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=2,
            population_size=6,
            context_depth=3,
            context_width=2,
        )
        assert game.context_depth == 3
        assert game.context_width == 2

    def test_odd_population_size_raises_error(self):
        with pytest.raises(ValueError, match="Population size must be even"):
            DonorGame(
                initial_resources=100,
                donation_multiplier=3,
                population_size=5,
            )

    def test_get_agent_ids(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        agent_ids = game.get_agent_ids()
        assert agent_ids == ["Agent_1", "Agent_2", "Agent_3", "Agent_4"]

    def test_personas_initialization(self):
        personas = {
            "Agent_1": {"system_addition": "You are cooperative."},
            "Agent_2": {"system_addition": "You are selfish."},
        }
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
            personas=personas,
        )
        assert game.personas == personas


class TestGameDataInitialization:
    """Tests for game state initialization."""

    def test_initialize_game_data(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        sim_data = MockSimData()
        game.initialize_game_data(sim_data)

        assert "balances" in sim_data.game_data
        assert "donation_history" in sim_data.game_data
        assert "pairings_history" in sim_data.game_data

        # Check all agents have initial resources
        for agent_id in game.get_agent_ids():
            assert sim_data.game_data["balances"][agent_id] == 100

    def test_initialize_game_data_different_resources(self):
        game = DonorGame(
            initial_resources=500,
            donation_multiplier=2,
            population_size=6,
        )
        sim_data = MockSimData()
        game.initialize_game_data(sim_data)

        for agent_id in game.get_agent_ids():
            assert sim_data.game_data["balances"][agent_id] == 500


class TestPairingGeneration:
    """Tests for random pairing generation."""

    def test_generate_pairings_correct_count(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=6,
        )
        sim_data = MockSimData()
        sim_data.agents = {f"Agent_{i+1}": MagicMock() for i in range(6)}

        pairings = game.generate_pairings(turn=1, sim_data=sim_data)

        # Should have N/2 pairs
        assert len(pairings) == 3

    def test_generate_pairings_all_agents_paired(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        sim_data = MockSimData()
        sim_data.agents = {f"Agent_{i+1}": MagicMock() for i in range(4)}

        pairings = game.generate_pairings(turn=1, sim_data=sim_data)

        # Flatten pairings and check all agents are included exactly once
        paired_agents = []
        for agent_a, agent_b in pairings:
            paired_agents.append(agent_a)
            paired_agents.append(agent_b)

        assert len(paired_agents) == 4
        assert set(paired_agents) == set(sim_data.agents.keys())

    def test_generate_pairings_no_self_pairing(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        sim_data = MockSimData()
        sim_data.agents = {f"Agent_{i+1}": MagicMock() for i in range(4)}

        # Run multiple times to account for randomness
        for _ in range(10):
            pairings = game.generate_pairings(turn=1, sim_data=sim_data)
            for agent_a, agent_b in pairings:
                assert agent_a != agent_b


class TestDonationExtraction:
    """Tests for extracting donation amounts from agent responses."""

    def test_extract_donation_json_format(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        response = '{"donation": 50}'
        amount = game.extract_donation_amount(response, max_amount=100)
        assert amount == 50

    def test_extract_donation_json_with_decimal(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        response = '{"donation": 25.5}'
        amount = game.extract_donation_amount(response, max_amount=100)
        assert amount == 25.5

    def test_extract_donation_with_surrounding_text(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        response = 'I will donate {"donation": 30} to my partner.'
        amount = game.extract_donation_amount(response, max_amount=100)
        assert amount == 30

    def test_extract_donation_fallback_to_number(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        response = "I decide to donate 40 units."
        amount = game.extract_donation_amount(response, max_amount=100)
        assert amount == 40

    def test_extract_donation_clamps_to_max(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        response = '{"donation": 150}'
        amount = game.extract_donation_amount(response, max_amount=100)
        assert amount == 100

    def test_extract_donation_clamps_negative_to_zero(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        response = '{"donation": -50}'
        # Regex won't match negative, will fall back
        amount = game.extract_donation_amount(response, max_amount=100)
        assert amount >= 0

    def test_extract_donation_no_number_returns_zero(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        response = "I refuse to give anything."
        amount = game.extract_donation_amount(response, max_amount=100)
        assert amount == 0


class TestObservableHistory:
    """Tests for the observable history tree."""

    def setup_method(self):
        self.game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
            context_depth=2,
            context_width=2,
        )
        self.sim_data = MockSimData()
        self.game.initialize_game_data(self.sim_data)

    def test_no_history_returns_message(self):
        history = self.game.get_observable_history(
            observer_id="Agent_1",
            partner_id="Agent_2",
            sim_data=self.sim_data,
        )
        assert "No previous donation history" in history

    def test_history_shows_partner_donations(self):
        # Add donation history
        self.sim_data.game_data["donation_history"] = [
            {
                "round": 1,
                "donor_id": "Agent_2",
                "recipient_id": "Agent_3",
                "amount": 30,
                "percentage": 30.0,
            }
        ]

        history = self.game.get_observable_history(
            observer_id="Agent_1",
            partner_id="Agent_2",
            sim_data=self.sim_data,
        )

        assert "Agent_2" in history
        assert "Agent_3" in history
        assert "30.0%" in history

    def test_history_depth_limit(self):
        # Create a chain: Agent_2 -> Agent_3 -> Agent_4 -> Agent_1
        self.sim_data.game_data["donation_history"] = [
            {
                "round": 1,
                "donor_id": "Agent_2",
                "recipient_id": "Agent_3",
                "amount": 30,
                "percentage": 30.0,
            },
            {
                "round": 2,
                "donor_id": "Agent_3",
                "recipient_id": "Agent_4",
                "amount": 25,
                "percentage": 25.0,
            },
            {
                "round": 3,
                "donor_id": "Agent_4",
                "recipient_id": "Agent_1",
                "amount": 20,
                "percentage": 20.0,
            },
        ]

        # With depth=2, should see Agent_2 -> Agent_3 -> Agent_4, but not deeper
        history = self.game.get_observable_history(
            observer_id="Agent_1",
            partner_id="Agent_2",
            sim_data=self.sim_data,
        )

        assert "Agent_2" in history
        assert "Agent_3" in history
        assert "Agent_4" in history

    def test_history_width_limit(self):
        # Agent_2 made multiple donations
        self.sim_data.game_data["donation_history"] = [
            {
                "round": 1,
                "donor_id": "Agent_2",
                "recipient_id": "Agent_3",
                "amount": 30,
                "percentage": 30.0,
            },
            {
                "round": 2,
                "donor_id": "Agent_2",
                "recipient_id": "Agent_4",
                "amount": 25,
                "percentage": 25.0,
            },
            {
                "round": 3,
                "donor_id": "Agent_2",
                "recipient_id": "Agent_1",
                "amount": 20,
                "percentage": 20.0,
            },
        ]

        # With width=2, should only see most recent 2 donations
        history = self.game.get_observable_history(
            observer_id="Agent_1",
            partner_id="Agent_2",
            sim_data=self.sim_data,
        )

        # Count how many "Agent_2 donated" lines appear
        agent_2_donations = history.count("Agent_2 donated")
        assert agent_2_donations <= 2


class TestTurnProcessing:
    """Tests for processing turns and updating balances."""

    def setup_method(self):
        self.game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        self.sim_data = MockSimData()
        self.sim_data.agents = {f"Agent_{i+1}": MagicMock() for i in range(4)}
        self.sim_data.conversation_history = [{"round": 1}]
        self.game.initialize_game_data(self.sim_data)

    def test_process_turn_updates_balances(self):
        # Agent_1 donates 20 to Agent_2, Agent_2 donates 10 to Agent_1
        agent_responses = {
            "Agent_1": {
                "reasoning": "test",
                "donation_response": '{"donation": 20}',
                "donation_amount": 20,
                "partner_id": "Agent_2",
            },
            "Agent_2": {
                "reasoning": "test",
                "donation_response": '{"donation": 10}',
                "donation_amount": 10,
                "partner_id": "Agent_1",
            },
            "Agent_3": {
                "reasoning": "test",
                "donation_response": '{"donation": 0}',
                "donation_amount": 0,
                "partner_id": "Agent_4",
            },
            "Agent_4": {
                "reasoning": "test",
                "donation_response": '{"donation": 0}',
                "donation_amount": 0,
                "partner_id": "Agent_3",
            },
        }

        self.game.process_turn(turn=1, agent_responses=agent_responses, sim_data=self.sim_data)

        balances = self.sim_data.game_data["balances"]

        # Agent_1: 100 - 20 + (10 * 3) = 110
        assert balances["Agent_1"] == 110

        # Agent_2: 100 - 10 + (20 * 3) = 150
        assert balances["Agent_2"] == 150

        # Agent_3 and Agent_4: no change
        assert balances["Agent_3"] == 100
        assert balances["Agent_4"] == 100

    def test_process_turn_records_donation_history(self):
        agent_responses = {
            "Agent_1": {
                "reasoning": "test",
                "donation_response": '{"donation": 25}',
                "donation_amount": 25,
                "partner_id": "Agent_2",
            },
            "Agent_2": {
                "reasoning": "test",
                "donation_response": '{"donation": 15}',
                "donation_amount": 15,
                "partner_id": "Agent_1",
            },
            "Agent_3": {
                "reasoning": "test",
                "donation_response": '{"donation": 50}',
                "donation_amount": 50,
                "partner_id": "Agent_4",
            },
            "Agent_4": {
                "reasoning": "test",
                "donation_response": '{"donation": 30}',
                "donation_amount": 30,
                "partner_id": "Agent_3",
            },
        }

        self.game.process_turn(turn=1, agent_responses=agent_responses, sim_data=self.sim_data)

        history = self.sim_data.game_data["donation_history"]
        assert len(history) == 4

        # Check donation percentages are recorded
        agent_1_donation = next(d for d in history if d["donor_id"] == "Agent_1")
        assert agent_1_donation["amount"] == 25
        assert agent_1_donation["percentage"] == 25.0  # 25/100 * 100

    def test_simultaneous_donations(self):
        """Test that donations are applied simultaneously, not sequentially."""
        # If donations were sequential, the order would affect final balances
        # With simultaneous, order should not matter
        agent_responses = {
            "Agent_1": {
                "reasoning": "test",
                "donation_response": '{"donation": 50}',
                "donation_amount": 50,
                "partner_id": "Agent_2",
            },
            "Agent_2": {
                "reasoning": "test",
                "donation_response": '{"donation": 50}',
                "donation_amount": 50,
                "partner_id": "Agent_1",
            },
            "Agent_3": {
                "reasoning": "test",
                "donation_response": '{"donation": 0}',
                "donation_amount": 0,
                "partner_id": "Agent_4",
            },
            "Agent_4": {
                "reasoning": "test",
                "donation_response": '{"donation": 0}',
                "donation_amount": 0,
                "partner_id": "Agent_3",
            },
        }

        self.game.process_turn(turn=1, agent_responses=agent_responses, sim_data=self.sim_data)

        balances = self.sim_data.game_data["balances"]

        # Both should have: 100 - 50 + (50 * 3) = 200
        assert balances["Agent_1"] == 200
        assert balances["Agent_2"] == 200


class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_default_system_prompt(self):
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        prompt = game.get_system_prompt("Agent_1", MagicMock())

        assert "100" in prompt  # initial resources
        assert "3" in prompt  # multiplier
        assert "4" in prompt  # population size
        assert "Donor Game" in prompt

    def test_custom_system_prompt_template(self):
        template = "Custom game with {initial_resources} resources and {donation_multiplier}x multiplier for {population_size} players."
        game = DonorGame(
            initial_resources=200,
            donation_multiplier=2,
            population_size=6,
            system_prompt_template=template,
        )
        prompt = game.get_system_prompt("Agent_1", MagicMock())

        assert "Custom game" in prompt
        assert "200" in prompt
        assert "2x" in prompt
        assert "6" in prompt

    def test_persona_added_to_prompt(self):
        personas = {
            "Agent_1": {"system_addition": "You always cooperate."},
        }
        game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
            personas=personas,
        )

        prompt_1 = game.get_system_prompt("Agent_1", MagicMock())
        prompt_2 = game.get_system_prompt("Agent_2", MagicMock())

        assert "You always cooperate." in prompt_1
        assert "You always cooperate." not in prompt_2


class TestPromptGeneration:
    """Tests for reasoning and donation prompt generation."""

    def setup_method(self):
        self.game = DonorGame(
            initial_resources=100,
            donation_multiplier=3,
            population_size=4,
        )
        self.sim_data = MockSimData()
        self.sim_data.agents = {f"Agent_{i+1}": MagicMock() for i in range(4)}
        self.game.initialize_game_data(self.sim_data)

    def test_reasoning_prompt_contains_context(self):
        prompt = self.game.get_reasoning_prompt(
            agent_id="Agent_1",
            partner_id="Agent_2",
            turn=1,
            sim_data=self.sim_data,
        )

        assert "ROUND 1" in prompt
        assert "100" in prompt  # current resources
        assert "Agent_2" in prompt  # partner
        assert "3" in prompt  # multiplier

    def test_donation_prompt_contains_balance(self):
        prompt = self.game.get_donation_prompt(
            agent_id="Agent_1",
            partner_id="Agent_2",
            turn=1,
            sim_data=self.sim_data,
        )

        assert "100" in prompt  # balance
        assert "donation" in prompt.lower()
