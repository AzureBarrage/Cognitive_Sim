## Cognitive_Sim Architecture

This document describes the current runtime architecture and dataflow for the Cognitive_Sim project.

### High-level components

1. **Agent** ([`CognitiveAgent`](Cognitive_Sim/src/core/agent.py:9))
   - Decides what to do next: learn, review, or sleep.
   - Tracks energy economy and simple success/failure counters.

2. **Memory** ([`MemoryLayer`](Cognitive_Sim/src/core/memory_layer.py:1))
   - Stores memory *metadata* (stability, timestamps) in an index and memory *payloads* as per-memory blobs.
   - Implements an Ebbinghaus-style retention curve.
   - Maintains an internal priority queue so “what should I review next?” doesn’t require scanning all memories.

3. **Network** ([`CognitiveNetwork`](Cognitive_Sim/src/core/network.py:9))
   - Simple MLP "brain" with plasticity modulation.
   - Computes a meta-cognitive uncertainty signal (entropy of weights) on a schedule.

4. **Optimizer** ([`CognitiveOptimizer`](Cognitive_Sim/src/core/optimizer.py:6))
   - Adam wrapper that adapts the learning rate based on entropy (exploration) and stability (exploitation).

5. **Environment** ([`SimulationEnvironment`](Cognitive_Sim/src/environment/simulation_env.py:5))
   - Produces “flashcards” (input/target pairs) from a dataset.

6. **Interfaces**
   - CLI ([`cli`](Cognitive_Sim/src/main.py:92))
   - API ([`app`](Cognitive_Sim/src/api.py:64))

### Dataflow (CLI run)

CLI entrypoint ([`cli`](Cognitive_Sim/src/main.py:92)) creates a simulation ([`CognitiveSimulation`](Cognitive_Sim/src/main.py:25)):

1. **Bootstrapping**
   - Loads config via [`load_config()`](Cognitive_Sim/src/config.py:35)
   - Loads memory index/payloads via [`MemoryLayer.load_state()`](Cognitive_Sim/src/core/memory_layer.py:1) unless `--fresh` is passed
   - Loads network weights via [`CognitiveNetwork.load_weights()`](Cognitive_Sim/src/core/network.py:109) unless `--fresh` is passed

2. **Loop** ([`run_training_loop()`](Cognitive_Sim/src/main.py:49))
   - Agent chooses action via [`decide_strategy()`](Cognitive_Sim/src/core/agent.py:35)
     - `learn_new`: gets a flashcard from the environment and trains network
     - `review`: retrieves a due memory; if recalled, reinforces by training; if forgotten, pays compute penalty
     - `sleep`: consolidates and boosts stability of due memories, and restores energy

3. **Shutdown**
   - Saves memory + weights at end of loop
     - [`MemoryLayer.save_state()`](Cognitive_Sim/src/core/memory_layer.py:1)
     - [`CognitiveNetwork.save_weights()`](Cognitive_Sim/src/core/network.py:103)

### Dataflow (API)

The API uses FastAPI lifespan hooks ([`lifespan()`](Cognitive_Sim/src/api.py:38)):

* Startup
  - load config
  - load memory index
  - load network weights
  - create agent with config

* Shutdown
  - persist memory index/blobs
  - persist network weights

Endpoints:
* `POST /teach` → calls [`CognitiveAgent.learn_new()`](Cognitive_Sim/src/core/agent.py:58)
* `POST /ask` → optionally retrieves memory context then does a forward pass
* `POST /sleep` → forces persistence
* `POST /act` → forces `sleep` or `review` for demo/testing

### Persistence format

Memory uses a two-level persistence strategy:

* **Index**: JSON at `memory.index_path` (default `data/memory_index.json`)
  - stores: `created_at`, `last_access`, `stability`, `access_count`, `data_path`
* **Payloads**: one file per memory at `memory.store_dir/<memory_id>.pt`
  - written with `torch.save` after moving tensors to CPU

Legacy compatibility:
* If `data/memory_store.json` exists (older metadata-only file), it will be loaded as metadata-only and payloads will remain missing until the system re-learns content.

### Scheduling / review selection

`MemoryLayer` computes next-review timestamps and maintains a min-heap. This allows:

* fast “is anything due?” checks
* bounded retrieval of due IDs

This is critical for scaling memory counts without turning each agent step into an O(N) scan.

### Configuration

Configuration is defined by Pydantic models in [`SimulationConfig`](Cognitive_Sim/src/config.py:18) and YAML environment files:

* Development: [`development.yaml`](Cognitive_Sim/configs/development.yaml:1)
* Production: [`production.yaml`](Cognitive_Sim/configs/production.yaml:1)
* Testing: [`testing.yaml`](Cognitive_Sim/configs/testing.yaml:1)
