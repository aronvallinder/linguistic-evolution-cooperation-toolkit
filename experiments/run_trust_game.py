from src.simulation import run_simulation
from src.myth_writer import MythWriter
from games.trust_game import TrustGame


if __name__ == "__main__":
    # Configure game
    game = TrustGame(endowment=5, multiplier=3)
    myth_writer = MythWriter(myth_topic="")
    model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    
    # Run simulation
    sim_data = run_simulation(
        game,
        model,
        temperature=0.8,
        num_turns=10,
        num_agents=2,
        memory_capacity=3,
        agent_biases="",
        myth_writer=myth_writer
    )
    
    # Save results
    base_path = "data/json/main_loop/"
    sim_data.save_state(base_path + "config_test.json")
    print(f"\n✓ Simulation state saved to config_test.json")