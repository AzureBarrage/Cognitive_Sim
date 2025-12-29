# Cognitive Simulation (Cognitive_Sim)

A biologically-inspired cognitive architecture that simulates an autonomous agent with memory constraints, energy management, and learning capabilities. The system uses a neural network as its "brain" and manages knowledge retention using principles of **Ebbinghaus decay** and **spaced repetition**.

## 🧠 Project Overview

Cognitive_Sim models the tension between acquiring new knowledge and maintaining existing memories. The agent must balance its limited **energy** reserves between two primary strategies:
1.  **Learn New**: Acquire new patterns from the environment (high energy cost).
2.  **Review**: Reinforce existing memories to prevent decay (moderate energy cost, energy reward for success).

If the agent neglects reviewing, memories become "unstable" and are eventually forgotten, incurring a high computational "re-learning" penalty.

## ✨ Key Features

*   **Agentic Decision Making**: The `CognitiveAgent` autonomously decides whether to learn or review based on energy levels and memory stability.
*   **Memory Decay System**: Implements realistic memory fading; memories must be reinforced to stay accessible.
*   **Energy Economy**: Actions have specific energy costs. Successful retrieval of memories provides energy rewards, simulating the biological imperative to maintain useful knowledge.
*   **Neural Core**: Uses **PyTorch** for the underlying cognitive network that learns and predicts patterns.
*   **Dual Interfaces**:
    *   **CLI Simulation**: Run automated training loops where the agent interacts with a data environment.
    *   **REST API**: A **FastAPI** interface to "teach" the agent, "ask" for predictions, and monitor its status in real-time.
*   **Stability Tracking**: Monitors system entropy and memory health to guide optimization.

## 🛠️ Installation

### Prerequisites
*   Python 3.8+
*   Docker (optional)

### Local Setup
1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Cognitive_Sim
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements/base.txt
    ```

3.  **Set PYTHONPATH** (Windows):
    ```bash
    set PYTHONPATH=.
    ```

### Docker Setup
Run the entire stack using Docker Compose:
```bash
docker-compose up --build
```

## 🚀 Usage

### 1. Command Line Interface (CLI)
Run a simulation loop where the agent interacts with the environment automatically.

```bash
# Run a 100-step simulation in the development environment
python src/main.py run --env development --steps 100

# Verify system components
python src/main.py verify
```

### 2. API Interface
Start the REST API server to interact with the agent manually.

```bash
python src/api.py
```

**Endpoints:**
*   `GET /status`: View agent's energy, memory count, and uptime.
*   `POST /teach`: Teach the agent a new pattern (input/target).
*   `POST /ask`: Ask the agent for a prediction (can specify a memory key for context).
*   `POST /sleep`: Force memory consolidation and save state.

## 📂 Project Structure

```
Cognitive_Sim/
├── configs/               # YAML configuration files (dev, prod, test)
├── src/
│   ├── api.py             # FastAPI REST endpoints
│   ├── main.py            # CLI entry point
│   ├── core/
│   │   ├── agent.py       # Core decision logic (Learn vs Review)
│   │   ├── memory_layer.py # Ebbinghaus decay with entropy calculations
│   │   ├── network.py     # PyTorch Neural Network
│   │   └── optimizer.py   # Custom optimizer
│   └── environment/       # Simulation environment and datasets
├── requirements/          # Project dependencies
└── docker-compose.yml     # Container orchestration
```

## ⚙️ Configuration

The system is configured via YAML files in the `configs/` directory. You can adjust parameters such as:
*   `network.learning_rate`: Base learning rate for the neural net.
*   `memory.decay_rate`: How fast memories fade.
*   `entropy_threshold`: The stability limit that triggers critical reviews.

## 🧪 Testing

Run the test suite to ensure system stability:
```bash
pytest tests/
```
