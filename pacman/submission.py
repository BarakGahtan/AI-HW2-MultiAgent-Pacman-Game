import random, util
from game import Agent, Actions, Directions
from util import manhattanDistance
import time

FOOD_FACTOR = 10
MOBILITY_FACTOR = 1
CAPSULE_FACTOR = 200
SCARED_GHOST_FACTOR = 200
NOT_SCARED_FACTOR = 50
CAPSULE_NUM_FACTOR = 2000
WINDOW = 4
BONUS_FACTOR = 2


#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None
    self.totalTime=0
    self.count=0

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
    startTime = time.time()

    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    endTime = time.time()
    self.totalTime += (endTime - startTime)
    self.count += 1

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    # return scoreEvaluationFunction(successorGameState)
    # TODO: new heuristic
    return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
  return gameState.getScore()

######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
  """


  The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

  A GameState specifies the full game state, including the food, capsules, agent configurations and more.
  Following are a few of the helper methods that you can use to query a GameState object to gather information about
  the present state of Pac-Man, the ghosts and the maze:

  gameState.getLegalActions():
  gameState.getPacmanState():
  gameState.getGhostStates():
  gameState.getNumAgents():
  gameState.getScore():
  The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
  """
  food_positions = gameState.getFood().asList()
  food_distances = list(map((lambda x: manhattanDistance(x, gameState.getPacmanPosition())), food_positions))
  min_food_dist = min(food_distances, default=1)
  food_factor = FOOD_FACTOR * (1 / min_food_dist)

  # mobility = 0
  mobility = len(gameState.getLegalActions())

  capsules_positions = gameState.getCapsules()
  capsules_dist = [manhattanDistance(c, gameState.getPacmanPosition()) for c in capsules_positions]
  min_capsule_dist = min(capsules_dist, default=0)
  capslue_dist_factor = CAPSULE_FACTOR * (1 / min_capsule_dist) if min_capsule_dist > 0 else 0
  num_of_capsulses = len(gameState.getCapsules())
  capslue_num_factor = CAPSULE_NUM_FACTOR * (1 / num_of_capsulses) if num_of_capsulses > 0 else 2*CAPSULE_NUM_FACTOR
  capslue_factor = capslue_dist_factor + capslue_num_factor

  # distance from ghost, we take into consideration if the ghost is scared or not. if it is scared, then
  # we want to get closer to it, and the opposite.
  distances_from_scared = []
  distances_from_not_scared = []
  for i in range(gameState.getNumAgents()):
    if i==0:
      continue
    else:
      ghost_state = gameState.getGhostState(i)
      pos = gameState.getGhostPosition(i)
      is_scared = ghost_state.scaredTimer > 0
      dist_from_ghost = manhattanDistance(pos, gameState.getPacmanPosition())
      if is_scared:
        distances_from_scared.append(dist_from_ghost)
      else:
        distances_from_not_scared.append(dist_from_ghost)

  min_scared_ghost_dist = min(distances_from_scared, default=0)
  scared_factor = SCARED_GHOST_FACTOR * (1 / min_scared_ghost_dist) if min_scared_ghost_dist > 0 else 0

  min_not_scared_ghost_dist = min(distances_from_not_scared, default=0)
  if min_not_scared_ghost_dist > WINDOW:
      not_scared_factor = NOT_SCARED_FACTOR * WINDOW
  else:
      not_scared_factor = NOT_SCARED_FACTOR * min_not_scared_ghost_dist

  return 5*gameState.getScore() + food_factor + MOBILITY_FACTOR * mobility + capslue_factor \
         + not_scared_factor + scared_factor



#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)
    self.totalTime = 0
    self.count = 0

######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE
    startTime = time.time()

    result = self.minimax(gameState, self.index, self.depth)[1]

    endTime = time.time()
    self.totalTime += (endTime - startTime)
    self.count += 1

    return result
    # END_YOUR_CODE


  def minimax(self, gameState, agent, depth):
    # Check if final state:
    if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions()) == 0:
      return self.evaluationFunction(gameState), None

    # Check if depth is 0:
    if depth == 0:
      return self.evaluationFunction(gameState), None

    next_agent = (agent + 1) % gameState.getNumAgents()
    if next_agent == 0:
      depth -= 1

    if agent == 0:
      # it's pacmans turn- calc max
      curr_max = -1e8
      action_to_return = None
      for action in gameState.getLegalActions(agent):
        result = self.minimax(gameState.generateSuccessor(agent, action), next_agent, depth)
        if result[0] > curr_max:
          curr_max = result[0]
          action_to_return = action

      return curr_max, action_to_return

    else:
      # its a ghost turn - calc min
      curr_min = 1e8
      action_to_return = None
      for action in gameState.getLegalActions(agent):
        result = self.minimax(gameState.generateSuccessor(agent, action), next_agent, depth)
        if result[0] < curr_min:
          curr_min = result[0]
          action_to_return = action

      return curr_min, action_to_return



######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """


    # BEGIN_YOUR_CODE
    startTime = time.time()

    result = self.alphaBeta(gameState, self.index, self.depth, -1e8, 1e8)[1]

    endTime = time.time()
    self.totalTime += (endTime - startTime)
    self.count += 1
    return result
    # END_YOUR_CODE

  def alphaBeta(self, gameState, agent, depth, alpha, beta):
    # Check if final state:
    if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions()) == 0:
      return self.evaluationFunction(gameState), None

    # Check if depth is 0:
    if depth == 0:
      return self.evaluationFunction(gameState), None

    next_agent = (agent + 1) % gameState.getNumAgents()
    if next_agent == 0:
      depth -= 1

    if agent == 0:
      # it's pacmans turn- calc max
      curr_max = -1e8
      action_to_return = None
      for action in gameState.getLegalActions(agent):
        result = self.alphaBeta(gameState.generateSuccessor(agent, action), next_agent, depth, alpha, beta)
        if result[0] > curr_max:
          curr_max = result[0]
          action_to_return = action
          alpha = max(curr_max, alpha)
          if curr_max >= beta:
            return curr_max, action_to_return  # TODO(Check why this is the right value to return)

      return curr_max, action_to_return

    else:
      # its a ghost turn - calc min
      curr_min = 1e8
      action_to_return = None
      for action in gameState.getLegalActions(agent):
        result = self.alphaBeta(gameState.generateSuccessor(agent, action), next_agent, depth, alpha, beta)
        if result[0] < curr_min:
          curr_min = result[0]
          action_to_return = action
          beta = min(curr_min, beta)
          if curr_min <= alpha:
            return curr_min, action_to_return  # TODO(Check why this is the right value to return)

      return curr_min, action_to_return

######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """

    # BEGIN_YOUR_CODE
    startTime = time.time()

    result = self.expectimax(gameState, self.index, self.depth)[1]

    endTime = time.time()
    self.totalTime += (endTime - startTime)
    self.count += 1
    return result
    # END_YOUR_CODE

  def expectimax(self, gameState, agent, depth):
    # Check if final state:
    if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions()) == 0:
      return self.evaluationFunction(gameState), None

    # Check if depth is 0:
    if depth == 0:
      return self.evaluationFunction(gameState), None

    next_agent = (agent + 1) % gameState.getNumAgents()
    if next_agent == 0:
      depth -= 1

    if agent == 0:
      # it's pacmans turn- calc max
      curr_max = -1e8
      action_to_return = None
      for action in gameState.getLegalActions(agent):
        result = self.expectimax(gameState.generateSuccessor(agent, action), next_agent, depth)
        if result[0] > curr_max:
          curr_max = result[0]
          action_to_return = action

      return curr_max, action_to_return

    else:
      number_of_possible_actions = len(gameState.getLegalActions(agent))
      probability_sum=0
      for action in gameState.getLegalActions(agent):
        result = self.expectimax(gameState.generateSuccessor(agent, action), next_agent, depth)
        probability = result[0]/number_of_possible_actions
        probability_sum += probability

      return probability_sum, None

######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

    # BEGIN_YOUR_CODE
    startTime = time.time()

    result = self.directionalExpectimax(gameState, self.index, self.depth)[1]

    endTime = time.time()
    self.totalTime += (endTime - startTime)
    self.count += 1
    return result
    # END_YOUR_CODE

  def directionalExpectimax(self, gameState, agent, depth):
    # Check if final state:
    if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions()) == 0:
      return self.evaluationFunction(gameState), None

    # Check if depth is 0:
    if depth == 0:
      return self.evaluationFunction(gameState), None

    next_agent = (agent + 1) % gameState.getNumAgents()
    if next_agent == 0:
      depth -= 1

    if agent == 0:
      # it's pacmans turn- calc max
      curr_max = -1e8
      action_to_return = None
      for action in gameState.getLegalActions(agent):
        result = self.directionalExpectimax(gameState.generateSuccessor(agent, action), next_agent, depth)
        if result[0] > curr_max:
          curr_max = result[0]
          action_to_return = action

      return curr_max, action_to_return

    else:
      probability_sum = 0
      distribution = self.getDistribution(gameState, agent)
      for action in gameState.getLegalActions(agent):
        result = self.directionalExpectimax(gameState.generateSuccessor(agent, action), next_agent, depth)
        probability = result[0] * distribution[action]
        probability_sum += probability
      return probability_sum, None

  def getDistribution(self, gameState, ghost_index):
    ''' this function is copied from DirectionalGhost class'''

    # Read variables from state
    ghostState = gameState.getGhostState(ghost_index)
    legalActions = gameState.getLegalActions(ghost_index)
    pos = gameState.getGhostPosition(ghost_index)
    isScared = ghostState.scaredTimer > 0

    speed = 1
    if isScared: speed = 0.5

    actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
    newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
    pacmanPosition = gameState.getPacmanPosition()

    # Select best actions given the state
    distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]
    if isScared:
      bestScore = max(distancesToPacman)
      bestProb = 0.2 # TODO: where do we get it from?
    else:
      bestScore = min(distancesToPacman)
      bestProb = 0.8  # TODO: where do we get it from?
    bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

    # Construct distribution
    dist = util.Counter()
    for a in bestActions: dist[a] = bestProb / len(bestActions)
    for a in legalActions: dist[a] += (1 - bestProb) / len(legalActions)
    dist.normalize()
    return dist

######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):

   def getAction(self, gameState):
     walls=gameState.getWalls()
     board_size = walls.width*walls.height
     if board_size>50:
       RANDOM_ROUND=45
       i = random.randint(1, RANDOM_ROUND)
       if (i % RANDOM_ROUND != 0):
         result = self.alphaBeta(gameState, self.index, 2, -1e8, 1e8)[1]
         return result
       else:
         legalMoves = gameState.getLegalActions()
         chosenIndex = random.choice(range(len(legalMoves)))  # Pick randomly
         return legalMoves[chosenIndex]
     else:
       result = self.expectimax(gameState, self.index, 3)[1]
       return result

   def alphaBeta(self, gameState, agent, depth, alpha, beta):
     # Check if final state:
     if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions()) == 0:
       return CompetitionFunction(gameState), None

     # Check if depth is 0:
     if depth == 0:
       return CompetitionFunction(gameState), None

     next_agent = (agent + 1) % gameState.getNumAgents()
     if next_agent == 0:
       depth -= 1

     if agent == 0:
       # it's pacmans turn- calc max
       curr_max = -1e8
       action_to_return = None

       shuffled = sorted(gameState.getLegalActions(agent), key=lambda L: random.random())
       for action in shuffled:
         result = self.alphaBeta(gameState.generateSuccessor(agent, action), next_agent, depth, alpha, beta)
         if result[0] > curr_max:
           curr_max = result[0]
           action_to_return = action
           alpha = max(curr_max, alpha)
           if curr_max >= beta:
             return curr_max, action_to_return

       return curr_max, action_to_return

     else:
       # its a ghost turn - calc min
       curr_min = 1e8
       action_to_return = None


       shuffled = sorted(gameState.getLegalActions(agent), key=lambda L: random.random())
       for action in shuffled:
         result = self.alphaBeta(gameState.generateSuccessor(agent, action), next_agent, depth, alpha, beta)
         if result[0] < curr_min:
           curr_min = result[0]
           action_to_return = action
           beta = min(curr_min, beta)
           if curr_min <= alpha:
             return curr_min, action_to_return

       return curr_min, action_to_return

   def expectimax(self, gameState, agent, depth):
    # Check if final state:
    if gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions()) == 0:
      return CompetitionFunction(gameState), None

    # Check if depth is 0:
    if depth == 0:
      return CompetitionFunction(gameState), None

    next_agent = (agent + 1) % gameState.getNumAgents()
    if next_agent == 0:
      depth -= 1

    if agent == 0:
      # it's pacmans turn- calc max
      curr_max = -1e8
      action_to_return = None
      shuffled = sorted(gameState.getLegalActions(agent), key=lambda L: random.random())
      for action in shuffled:
        result = self.expectimax(gameState.generateSuccessor(agent, action), next_agent, depth)
        if result[0] > curr_max:
          curr_max = result[0]
          action_to_return = action

      return curr_max, action_to_return

    else:
      number_of_possible_actions = len(gameState.getLegalActions(agent))
      probability_sum=0
      shuffled = sorted(gameState.getLegalActions(agent), key=lambda L: random.random())
      for action in shuffled:
        result = self.expectimax(gameState.generateSuccessor(agent, action), next_agent, depth)
        probability = result[0]/number_of_possible_actions
        probability_sum += probability

      return probability_sum, None

def CompetitionFunction(gameState):

 pacman_position = gameState.getPacmanPosition()
 food_positions = gameState.getFood().asList()
 food_distances = list(map((lambda x: manhattanDistance(x, pacman_position)), food_positions))
 min_food_dist = min(food_distances, default=1)
 food_factor = FOOD_FACTOR * (1 / min_food_dist)

 capsules_positions = gameState.getCapsules()
 capsules_dist = [manhattanDistance(c, pacman_position) for c in capsules_positions]
 min_capsule_dist = min(capsules_dist, default=0)
 capslue_dist_factor = CAPSULE_FACTOR * (1 / min_capsule_dist) if min_capsule_dist > 0 else 0
 num_of_capsulses = len(capsules_positions)
 capslue_num_factor = CAPSULE_NUM_FACTOR * (
           1 / num_of_capsulses) if num_of_capsulses > 0 else 2 * CAPSULE_NUM_FACTOR
 capslue_factor = capslue_dist_factor + capslue_num_factor

 distances_from_scared = []
 distances_from_not_scared = []
 for i in range(gameState.getNumAgents()):
   if i == 0:
     continue
   else:
     ghost_state = gameState.getGhostState(i)
     pos = gameState.getGhostPosition(i)
     is_scared = ghost_state.scaredTimer > 2
     dist_from_ghost = manhattanDistance(pos, pacman_position)
     if is_scared:
       distances_from_scared.append(dist_from_ghost)
     else:
       distances_from_not_scared.append(dist_from_ghost)

 min_scared_ghost_dist = min(distances_from_scared, default=0)
 scared_factor = SCARED_GHOST_FACTOR * (1 / min_scared_ghost_dist) if min_scared_ghost_dist > 0 else 0

 min_not_scared_ghost_dist = min(distances_from_not_scared, default=0)
 if min_not_scared_ghost_dist > WINDOW:
   not_scared_factor = NOT_SCARED_FACTOR * WINDOW
 else:
   not_scared_factor = NOT_SCARED_FACTOR * min_not_scared_ghost_dist

 return 5 * gameState.getScore() + food_factor + capslue_factor + not_scared_factor + scared_factor

