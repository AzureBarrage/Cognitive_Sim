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
from src.utils.metrics import MetricsTracker
from src.system_health import SystemHealth
from src.utils.visualizer import Visualizer

class CognitiveSimulation:
    def __init__(self, config_env: str = "development", fresh: bool = False):
        self.config = load_config(config_env)
        
        # Initialize Core Components
        self.memory = MemoryLayer(self.config.memory)
        self.network = CognitiveNetwork(self.config.network)
        self.optimizer = CognitiveOptimizer(self.network.parameters(), 
                                          base_lr=self.config.network.learning_rate)
        
        # Load persisted state unless starting fresh
        if not fresh:
            self.memory.load_state()
            self.network.load_weights("data/network_weights.pth")

        # Initialize Agent
        self.agent = CognitiveAgent(self.memory, self.network, self.optimizer, config=self.config.agent)
        
        # Initialize Environment
        self.data_manager = DatasetManager(self.config.dict())
        self.data_manager.load_data()
        self.env = SimulationEnvironment(self.data_manager.get_train_loader(batch_size=1))
        
        # Initialize Monitoring
        self.tracker = StabilityTracker(self.config.entropy_threshold)
        self.entropy_calc = EntropyCalculator()

        self.metrics = MetricsTracker()
        self.health = SystemHealth()
        self.visualizer = Visualizer()
        
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

                entropy = float(self.network.calculate_uncertainty())
                self.metrics.log_training_step(
                    step=step,
                    loss=float(result.get('loss', 0.0) or 0.0),
                    entropy=entropy,
                    plasticity=float(self.network.plasticity.mean().item())
                )
            
            elif strategy == "review":
                result = self.agent.review()
                print(f"Action: Review | Status: {result['status']}")
                if 'energy_gained' in result:
                    print(f"  -> Energy Gained: +{result['energy_gained']}")
                if 'penalty' in result:
                    print(f"  -> Penalty: {result['penalty']}")

            elif strategy == "sleep":
                result = self.agent.sleep()
                print(f"Action: Sleep | Status: {result['status']} | Boosted: {result.get('boosted_memories', 0)}")

            # System health
            usage = self.health.monitor.get_resource_usage()
            self.metrics.log_system_health(step=step, memory_usage=float(usage.get('memory_rss_mb', 0.0)), cpu_usage=float(usage.get('cpu_percent', 0.0)))
            
            time.sleep(0.05)  # Simulate processing time

        # Persist + plot at end of run
        self.memory.save_state()
        self.network.save_weights("data/network_weights.pth")
        self.visualizer.plot_training_history(self.metrics.get_history("training"))

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
@click.option('--fresh', is_flag=True, help='Start without loading persisted memory/weights')
def run(env, steps, fresh):
    """Run the cognitive simulation."""
    sim = CognitiveSimulation(env, fresh=fresh)
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
