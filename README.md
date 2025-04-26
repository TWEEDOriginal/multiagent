# Mini Project 1: Multi-Agent Search – COMP 5316

## Overview

This project extends the classic Pacman game by introducing multi-agent decision-making. The main goal is to implement intelligent agents capable of navigating Pacman's world while avoiding ghosts and collecting food efficiently. This involves reflex-based strategies and search algorithms such as Minimax, Alpha-Beta Pruning, and Expectimax, as well as designing effective evaluation functions.

The core logic was implemented in the `multiAgents.py` file, with agents handling different aspects of the game using adversarial search strategies.

---

## Implemented Questions and Descriptions

### **Q1: Reflex Agent**
- **File**: `multiAgents.py`
- **Modifications**:
  - Enhanced the evaluation function to account for:
    - Distance to the nearest food (using reciprocal to prioritize closer food).
    - Proximity to ghosts, applying a heavy penalty if a ghost is too close and not scared.
    - Bonus if ghosts are scared and within reach.
  - Goal: Guide Pacman to eat food while avoiding imminent threats.

### **Q2: Minimax Agent**
- **File**: `multiAgents.py`
- **Modifications**:
  - Implemented the `MinimaxAgent` using a recursive minimax search tree.
  - Each level alternates between Pacman (maximizer) and ghosts (minimizers).
  - Correctly handles multiple ghosts and arbitrary depths.
  - Uses the provided `self.evaluationFunction` to evaluate terminal states.

### **Q3: Alpha-Beta Agent**
- **File**: `multiAgents.py`
- **Modifications**:
  - Extended the MinimaxAgent with alpha-beta pruning to eliminate branches that cannot affect the final decision.
  - Ensured pruning logic supports multiple ghost agents and maintains correct node exploration count.
  - Maintained original child-processing order to match autograder expectations.

### **Q4: Expectimax Agent**
- **File**: `multiAgents.py`
- **Modifications**:
  - Implemented `ExpectimaxAgent` for environments where ghosts behave stochastically.
  - Instead of minimizing, ghost agents now return the expected utility (average of all outcomes).
  - Suitable for modeling random or probabilistic opponents.

### **Q5: Evaluation Function**
- **File**: `multiAgents.py` (`betterEvaluationFunction`)
- **Modifications**:
  - Designed a heuristic to evaluate game states based on:
    - Score from food collected.
    - Distance to the nearest food.
    - Distance to active and scared ghosts.
    - Number of remaining capsules and food.
    - Balanced aggressiveness when ghosts are scared with cautiousness when they are active.

---

## How to Run

You can test each agent using the following commands:

```bash
# Reflex Agent
python pacman.py -p ReflexAgent -l testClassic

# Minimax Agent
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=3

# Alpha-Beta Agent
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=3

# Expectimax Agent
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3

# Better Evaluation Function (used with an agent that accepts it)
python autograder.py -q q5
