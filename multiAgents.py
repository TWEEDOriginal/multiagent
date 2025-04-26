# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        ghostDistances = []
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            distance = manhattanDistance(newPos, ghostPos)

            # If the ghost is not scared add to distances
            if ghostState.scaredTimer == 0:
                ghostDistances.append(distance)

        if len(ghostDistances) > 0:
            closestGhost = min(ghostDistances)
            if closestGhost < 2:
                # one could lose the game if the ghost is too near
                closestGhost = -100
        else:
            # nothing can hurt you because ghosts are scared
            closestGhost = 100

        newFoodPositions = newFood.asList()

        foodDistances = [
            manhattanDistance(newPos, foodPosition) for foodPosition in newFoodPositions
        ]
        # no food for this action
        if len(foodDistances) == 0:
            # food plays no effect
            closestFood = 1 / 10
        else:
            closestFood = min(foodDistances)

        # Stop action would reduce score because of the pacman's timer constraint
        if action == Directions.STOP:
            score -= 50

        # the closer the food is, the more valuable it is
        foodScore = closestFood * 10
        # the closer the ghost is, the more dangerous it is
        return score + (closestGhost / foodScore)


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # Format of result = [score, action]
        result = self.value(gameState, 0, 0)

        # Return the action from result
        return result[1]

    def value(self, gameState: GameState, agentIndex, depth):
        """
        Returns value as pair of [score, action] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """

        # Terminal states:
        # end of game or final depth
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""

        # Max-agent: Pacman has index = 0
        if agentIndex == 0:
            return self.max(gameState, agentIndex, depth)

        # Min-agent: Ghost has index > 0
        else:
            return self.min(gameState, agentIndex, depth)

    def max(self, gameState: GameState, agentIndex, depth):
        """
        Returns the max utility action-score for pacman
        """
        v = float("-inf")
        chosenAction = ""
        legalMoves = gameState.getLegalActions(agentIndex)

        for action in legalMoves:
            # Generate the successor state after this action
            successorState = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth  # current depth

            # Pacman is next agent because current depth is done
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.value(successorState, successorIndex, successorDepth)[0]

            if successorValue > v:
                chosenAction = action
                v = successorValue

        return [v, chosenAction]

    def min(self, gameState: GameState, agentIndex, depth):
        """
        Returns the min utility action-score for ghosts
        """
        v = float("inf")
        chosenAction = ""
        legalMoves = gameState.getLegalActions(agentIndex)

        for action in legalMoves:
            # Generate the successor state after this action
            successorState = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth  # current depth

            # Pacman is next agent because current depth is done
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.value(successorState, successorIndex, successorDepth)[0]

            if successorValue < v:
                chosenAction = action
                v = successorValue

        return [v, chosenAction]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Format of result = [score, action]
        result = self.value(gameState, 0, 0, float("-inf"), float("inf"))

        # Return the action from result
        return result[1]

    def value(self, gameState: GameState, agentIndex, depth, alpha, beta):
        """
        Returns value as pair of [score, action] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """

        # Terminal states:
        # end of game or final depth
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""

        # Max-agent: Pacman has index = 0
        if agentIndex == 0:
            return self.max(gameState, agentIndex, depth, alpha, beta)

        # Min-agent: Ghost has index > 0
        else:
            return self.min(gameState, agentIndex, depth, alpha, beta)

    def max(self, gameState: GameState, agentIndex, depth, alpha, beta):
        """
        Returns the max utility action-score for pacman
        """
        v = float("-inf")
        chosenAction = ""
        legalMoves = gameState.getLegalActions(agentIndex)
        

        for action in legalMoves:
            # Generate the successor state after this action
            successorState = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth  # current depth

            # Pacman is next agent because current depth is done
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.value(successorState, successorIndex, successorDepth, alpha, beta)[0]

            if successorValue > v:
                chosenAction = action
                v = successorValue
            alpha = max(alpha, successorValue)
            # prune remaining actions (no pruning on equality as per instructions)
            if alpha > beta:
                return [v, chosenAction]   

        return [v, chosenAction]

    def min(self, gameState: GameState, agentIndex, depth, alpha, beta):
        """
        Returns the min utility action-score for ghosts
        """
        v = float("inf")
        chosenAction = ""
        legalMoves = gameState.getLegalActions(agentIndex)

        for action in legalMoves:
            # Generate the successor state after this action
            successorState = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth  # current depth

            # Pacman is next agent because current depth is done
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.value(successorState, successorIndex, successorDepth, alpha, beta)[0]

            if successorValue < v:
                chosenAction = action
                v = successorValue
            beta = min(beta, successorValue)
            # prune remaining actions (no pruning on equality as per instructions)
            if alpha > beta:
                return [v, chosenAction]   

        return [v, chosenAction]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        # Format of result = [score, action]
        result = self.value(gameState, 0, 0)

        # Return the action from result
        return result[1]

    def value(self, gameState: GameState, agentIndex, depth):
        """
        Returns value as pair of [score, action] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """

        # Terminal states:
        # end of game or final depth
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""

        # Max-agent: Pacman has index = 0
        if agentIndex == 0:
            return self.max(gameState, agentIndex, depth)

        # Min-agent: Ghost has index > 0
        else:
            return self.exp(gameState, agentIndex, depth)

    def max(self, gameState: GameState, agentIndex, depth):
        """
        Returns the max utility action-score for pacman
        """
        v = float("-inf")
        chosenAction = ""
        legalMoves = gameState.getLegalActions(agentIndex)

        for action in legalMoves:
            # Generate the successor state after this action
            successorState = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth  # current depth

            # Pacman is next agent because current depth is done
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.value(successorState, successorIndex, successorDepth)[0]

            if successorValue > v:
                chosenAction = action
                v = successorValue

        return [v, chosenAction]

    
    def exp(self, gameState: GameState, agentIndex, depth):
        """
        Returns the expected utility action-score for ghost
        """
        v = 0
        legalMoves = gameState.getLegalActions(agentIndex)
        # uniform probability 
        p = 1.0 / len(legalMoves)

        for action in legalMoves:
            # Generate the successor state after this action
            successorState = gameState.generateSuccessor(agentIndex, action)
            successorIndex = agentIndex + 1
            successorDepth = depth  # current depth

            # Pacman is next agent because current depth is done
            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            successorValue = self.value(successorState, successorIndex, successorDepth)[0]
            v += p * successorValue

        return [v, ""]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: returns the best score that allows it to 
    1) Start with the game's base score.
    2) Add a bonus that is larger when the nearest food is closer (1 / distance).
    3) Subtract a large penalty if non-scared ghosts are near.
    4) Slight bonus if ghosts are scared (you can chase/eat them).
    5) Light penalty if too many capsules or food remain.
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    pacmanPos = currentGameState.getPacmanPosition()
    # 1) Base Score
    score = currentGameState.getScore()

    # 2) Reward closeness to food
    foodPositions = currentGameState.getFood().asList()
    foodCount = len(foodPositions)
    if foodCount > 0:
        # Compute distance to the closest food
        minFoodDist = min(util.manhattanDistance(pacmanPos, fPos) for fPos in foodPositions)
        # Add a bonus: the closer the food, the bigger the reward.
        score += 1.0 / (minFoodDist)

    # 3) Account for ghost distances
    ghostStates = currentGameState.getGhostStates()
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDist = util.manhattanDistance(pacmanPos, ghostPos)
        # ghost is not scared:
        if ghost.scaredTimer == 0:
            # Penalize being too close to a dangerous ghost
            # For example, if ghostDist < 2, you might be in trouble.
            if ghostDist < 2:
                score -= 200  # Big penalty if you're about to get eaten
            else:
                # Some penalty that lessens as ghostDist grows
                score -= 5.0 / (ghostDist)
        else:
            # 4) If the ghost is scared, we could chase it for extra points
            # The closer we are, the more likely we can eat it
            score += 2.0 / (ghostDist)

    # 5) Capsules and leftover food
    # so the game prioritizes capsule hunting and escaping ghosts
    # A small penalty for each remaining capsule
    score -= 4 * len(currentGameState.getCapsules())
    # A small penalty for leftover food to encourage finishing
    score -= 1 * foodCount

    return score


# Abbreviation
better = betterEvaluationFunction
