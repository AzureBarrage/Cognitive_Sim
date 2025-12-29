from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import torch
import numpy as np
import time
from contextlib import asynccontextmanager

from src.config import load_config
from src.core.memory_layer import MemoryLayer
from src.core.network import CognitiveNetwork
from src.core.optimizer import CognitiveOptimizer
from src.core.agent import CognitiveAgent
from src.utils.logger import logger

# --- Data Models ---
class TeachRequest(BaseModel):
    input_data: List[float]
    target_data: List[float]
    label: Optional[str] = None # Optional human-readable tag

class PredictionRequest(BaseModel):
    input_data: List[float]
    memory_key: Optional[str] = None # Try to recall specific memory context

class AgentStatus(BaseModel):
    energy: float
    total_rewards: float
    memory_count: int
    at_risk_memories: int
    uptime_seconds: float

# --- Global State ---
sim_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing Cognitive System...")
    config = load_config("development")
    
    # Initialize Core
    memory = MemoryLayer(config.memory)
    memory.load_state() # Load persisted memories
    
    network = CognitiveNetwork(config.network)
    network.load_weights("data/network_weights.pth") # Load persisted brain
    
    optimizer = CognitiveOptimizer(network.parameters(), base_lr=config.network.learning_rate)
    agent = CognitiveAgent(memory, network, optimizer)
    
    sim_state['agent'] = agent
    sim_state['start_time'] = time.time()
    
    yield
    
    # Shutdown
    logger.info("Shutting down... Consolidating memories.")
    sim_state['agent'].memory.save_state()
    sim_state['agent'].network.save_weights("data/network_weights.pth")
    logger.info("System preserved.")

app = FastAPI(title="Cognitive Simulation API", lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "online", "message": "Cognitive Agent Ready. Time acts on us all."}

@app.get("/status", response_model=AgentStatus)
def get_status():
    agent = sim_state['agent']
    return AgentStatus(
        energy=agent.energy,
        total_rewards=agent.total_rewards,
        memory_count=len(agent.memory.memories),
        at_risk_memories=len(agent.memory.get_at_risk_memories()),
        uptime_seconds=time.time() - sim_state['start_time']
    )

@app.post("/teach")
def teach_agent(request: TeachRequest):
    """
    Teach the agent a new pattern. 
    This reinforces the memory and updates weights.
    """
    agent = sim_state['agent']
    
    # Convert list to tensor
    input_tensor = torch.tensor([request.input_data], dtype=torch.float32)
    target_tensor = torch.tensor([request.target_data], dtype=torch.float32)
    
    # Agent Logic
    result = agent.learn_new(input_tensor, target_tensor)
    
    return {
        "action": "learned",
        "loss": result.get('loss'),
        "energy_remaining": agent.energy,
        "memory_id": result.get("memory_id")
    }

@app.post("/ask")
def ask_agent(request: PredictionRequest):
    """
    Ask the agent to predict or recall.
    If 'memory_key' is provided, it tries to access that specific memory first.
    If memory is decayed, it might fail to retrieve context.
    """
    agent = sim_state['agent']
    input_tensor = torch.tensor([request.input_data], dtype=torch.float32)
    
    memory_context = None
    recall_status = "no_context_requested"
    
    # Try to retrieve context from Long Term Memory
    if request.memory_key:
        memory_data = agent.memory.retrieve_memory(request.memory_key)
        if memory_data:
            recall_status = "success"
            # Assuming stored data has 'input' tensor we can use as context
            # simplified for this template:
            memory_context = torch.tensor(memory_data['input'], dtype=torch.float32) if isinstance(memory_data, dict) else None
        else:
            recall_status = "forgotten"
    
    # Forward pass (Brain)
    with torch.no_grad():
        output, meta = agent.network(input_tensor, memory_context)
    
    return {
        "prediction": output.tolist()[0],
        "uncertainty": meta['uncertainty'],
        "recall_status": recall_status,
        "energy": agent.energy
    }

@app.post("/sleep")
def trigger_sleep():
    """
    Force a consolidation/save event.
    """
    sim_state['agent'].memory.save_state()
    sim_state['agent'].network.save_weights("data/network_weights.pth")
    return {"status": "slept", "message": "Memories consolidated."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
