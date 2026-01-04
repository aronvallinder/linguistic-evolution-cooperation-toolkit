# ============================================================================
# MYTH WRITING
# ============================================================================

class MythWriter:
    """Handles myth writing functionality, separate from game logic"""
    
    def __init__(self, myth_topic):
        self.myth_topic = myth_topic
    
    def get_myth_prompt_round_1(self, agent_id, turn, sim_data):
        """Generate prompt for myth writing"""
        return f"""Write a myth about {self.myth_topic}. Write 200 words."""

    def get_myth_prompt_round_later(self, agent_id, turn, sim_data):
        """Generate prompt for myth writing with the agent's previous myth"""
        # Get this agent's myth from previous round
        last_myth = ""
        other_agent_myth = ""
        
        for entry in sim_data.conversation_history:
            if entry["round"] == turn - 1 and "myths" in entry:
                if agent_id in entry["myths"]:
                    last_myth = entry["myths"][agent_id]
                
                # Get the other agent's myth
                for other_agent_id, myth in entry["myths"].items():
                    if other_agent_id != agent_id:
                        other_agent_myth = myth
                        break
                break
        
        if last_myth:
            if other_agent_myth:
                prompt = f"""Here is the myth you wrote in the previous round: 
{last_myth}

Here is the myth the other agent wrote in the previous round:
{other_agent_myth}

Write your own myth. Use your previous myth as inspiration, but adapt it in your own way. 
Write 200 words. Format exactly:
Myth: [your story here]. """
            else:
                raise ValueError(f"OTHER AGENT MYTH ERROR:No previous myth found for {agent_id} (other agent) in round {turn - 1}. Cannot generate later round prompt.")
        
        else:
            raise ValueError(f"NO SELF MYTH ERROR: No previous myth found for {agent_id} (you/self agent) in round {turn - 1}. Cannot generate later round prompt.")
        
        return prompt


    def process_myths(self, turn, agent_myths, sim_data):
        """Store the myths written by agents"""
        # Find the pre-created entry for this turn and fill in myths
        for entry in sim_data.conversation_history:
            if entry["round"] == turn:
                entry["myths"] = agent_myths
                break