# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
  A base class for reflex agents that chooses score-maximizing actions
  """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.Xmidpoint = ((gameState.getWalls().asList()[-1][0] + 1) / 2) - 1
        self.Ydist = gameState.getWalls().asList()[-1][1]


    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest Q(s,a).
    """
        actions = gameState.getLegalActions(self.index)
        print actions
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
    Finds the next successor which is a grid position (location tuple).
    """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
    Computes a linear combination of features and feature weights
    """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
    Returns a counter of features for the state
    """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
        return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def getFeatures(self, gameState, action):

        # initialize all features as zero
        features = util.Counter()
        # successor is a gameState object
        successor = self.getSuccessor(gameState, action)
        # list of all food coordinates - in tuple form - that our team needs to eat
        foodList = self.getFood(successor).asList()
        # list of all capsule coordinates - in tuple form - that our team can eat
        capsuleList = self.getCapsules(successor)
        # indices of opponents
        opponents = self.getOpponents(gameState)
        # position of our offensive agent after taking the given action
        pos = successor.getAgentPosition(self.index)

        # successorScore feature...idk what this does, it was in baselineTeam.
        # system breaks without it
        features['successorScore'] = -len(foodList) - len(capsuleList)  # self.getScore(successor)

        # ExactOpponentGhostDist feature...If we know the exact distance of either or both
        # opponents, then the feature is the distance to the nearest opponent
        distances = []
        for opp in opponents:
            observation = self.getCurrentObservation().getAgentState(opp).getPosition()
            if observation is not None:
                if observation[0] > self.Xmidpoint:
                    distances.append(self.getMazeDistance(observation, pos))
                    features['ExactOpponentGhostDist'] = min(distances)

        # distToNearestHome feature...the feature is the distance to the nearest safe zone (home)
        distances = []
        for i in range(1, self.Ydist - 1):
            if not gameState.hasWall(self.Xmidpoint, i):
                distances.append((self.getMazeDistance((self.Xmidpoint, i), pos), action))
        features['distToNearestHome'] = min(distances, key=lambda item: item[0])[0]

        # distToNearestFood feature... the feature is the distance to the nearest food
        # distToNearestFood2 feature... the feature is the distance to the 2nd nearest food
        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            sort = sorted([self.getMazeDistance(myPos, food) for food in foodList])
            minDistanceFood = sort[0]
            if len(sort) > 1:
                minDistanceFood2 = sort[1]
                features['distToNearestFood2'] = minDistanceFood2
            features['distToNearestFood'] = minDistanceFood

        # distToNearestCapsule feature... the feature is the distance to the nearest capsule
        if len(capsuleList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistanceCapsule = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
            features['distToNearestCapsule'] = minDistanceCapsule

        # numFoodCarry feature... this feature is the number of food that our pacman is carrying
        numCarrying = successor.getAgentState(self.index).numCarrying
        features['numFoodCarry'] = numCarrying

        return features

    def getWeights(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        if len(foodList) <= 2:
            return {'successorScore': 100, 'distToNearestFood': 0, 'distToNearestFood2': 0,
                'distToNearestCapsule': 0, 'ExactOpponentGhostDist': 20000, 'distToNearestHome': -7000,
                'numFoodCarry': 0}
        else:
            return {'successorScore': 1000, 'distToNearestFood': -3, 'distToNearestFood2': -2,
                    'distToNearestCapsule': -1, 'ExactOpponentGhostDist': 2000, 'distToNearestHome': 0,
                    'numFoodCarry': 0}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
