import sys
from pathlib import Path

# Add the project root to sys.path to resolve imports correctly
sys.path.append(str(Path(__file__).resolve().parent.parent))

import click
import torch
import numpy as np
import time
from typing import Optional

from src.config import load_config, SimulationConfig
from src.core.memory_layer import MemoryLayer
from src.core.network import CognitiveNetwork
from src.core.optimizer import CognitiveOptimizer
from src.core.entropy_calculator import EntropyCalculator
from src.environment.scheduler import CognitiveScheduler
from src.environment.stability_tracker import StabilityTracker
from src.environment.dataset_manager import DatasetManager
from src.environment.simulation_env import SimulationEnvironment
from src.core.agent import CognitiveAgent
from src.utils.logger import logger

class CognitiveSimulation:
    def __init__(self, config_env: str = "development"):
        self.config = load_config(config_env)
        
        # Initialize Core Components
        self.memory = MemoryLayer(self.config.memory)
        self.network = CognitiveNetwork(self.config.network)
        self.optimizer = CognitiveOptimizer(self.network.parameters(), 
                                          base_lr=self.config.network.learning_rate)
        
        # Initialize Agent
        self.agent = CognitiveAgent(self.memory, self.network, self.optimizer)
        
        # Initialize Environment
        self.data_manager = DatasetManager(self.config.dict())
        self.data_manager.load_data()
        self.env = SimulationEnvironment(self.data_manager.get_train_loader(batch_size=1))
        
        # Initialize Monitoring
        self.tracker = StabilityTracker(self.config.entropy_threshold)
        self.entropy_calc = EntropyCalculator()
        
        self.is_running = False

    def run_training_loop(self, steps: int = 100):
        """
        Run the Agentic Simulation Loop.
        """
        print(f"Starting Agentic Simulation with {steps} steps...")
        self.is_running = True
        
        for step in range(steps):
            print(f"\n--- Step {step+1}/{steps} ---")
            print(f"Energy: {self.agent.energy:.2f} | Rewards: {self.agent.total_rewards:.2f} | Cycles Spent: {self.agent.compute_cycles_spent:.2f}")
            
            # 1. Agent Decides Strategy
            strategy = self.agent.decide_strategy()
            
            if strategy == "learn_new":
                # Get input from Environment
                input_data, target = self.env.get_next_flashcard()
                result = self.agent.learn_new(input_data, target)
                print(f"Action: Learned New | Loss: {result.get('loss', 0):.4f} | Status: {result['status']}")
            
            elif strategy == "review":
                result = self.agent.review()
                print(f"Action: Review | Status: {result['status']}")
                if 'energy_gained' in result:
                    print(f"  -> Energy Gained: +{result['energy_gained']}")
                if 'penalty' in result:
                    print(f"  -> Penalty: {result['penalty']}")
            
            time.sleep(0.05)  # Simulate processing time

    def verify(self):
        """Verify system components."""
        try:
            print("[OK] Configuration loaded")
            print("[OK] Memory Layer initialized")
            print("[OK] Neural Network initialized")
            print("[OK] Optimizer initialized")
            print("[OK] Agent initialized")
            print("[OK] Environment initialized")
            print("System verification successful!")
        except Exception as e:
            print(f"[FAIL] System verification failed: {str(e)}")

@click.group()
def cli():
    pass

@cli.command()
@click.option('--env', default='development', help='Environment configuration to use')
@click.option('--steps', default=20, help='Number of simulation steps')
def run(env, steps):
    """Run the cognitive simulation."""
    sim = CognitiveSimulation(env)
    sim.run_training_loop(steps)

@cli.command()
def verify():
    """Verify system components."""
    try:
        sim = CognitiveSimulation()
        sim.verify()
    except Exception as e:
        print(f"[FAIL] System verification failed: {str(e)}")

if __name__ == "__main__":
    cli()
