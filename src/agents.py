from src.utils import call_llm

class Agent:
    """Represents a single LLM agent with memory capacity"""

    def __init__(self, agent_id, model, temperature, client, memory_capacity, initial_bias, system_prompt=None): #system prompt is none, so it can be set later. 
        self.agent_id = agent_id
        self.model = model
        self.temperature = temperature
        self.client = client
        self.memory_capacity = memory_capacity
        self.messages = []
        self.initial_bias = initial_bias
        self.system_prompt = system_prompt
    

    # Response with messages
    def respond(self, prompt):
        """Respond to a prompt with the LLM. The truncation effectuates
        a short term memory effect; earlier interactions are forgotten. Thus introduces recency bias; 
        recent interactions have more impact than older ones. Remove if unwanted"""
        # Truncate oldest messages if memory is full (but keep system prompt)
        if len(self.messages) > self.memory_capacity * 2 + 1:  # *2 for user and assistant messages, +1 for system prompt
            # Keep system prompt (first message) and last N messages
            self.messages = [self.messages[0]] + self.messages[-(self.memory_capacity * 2):]
        
        self.messages.append({"role": "user", "content": prompt})

        # Call the LLM
        response = call_llm(self.client, self.model, self.temperature, self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response