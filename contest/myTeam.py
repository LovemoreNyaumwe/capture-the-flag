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
        self.time = 0
        self.timeFlag = 0
        self.savedFood = None
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.team = "Red"
        if self.getTeam(gameState)[0] % 2 == 0:
            self.Xmidpoint = ((gameState.getWalls().asList()[-1][0] + 1) / 2) - 1
        else:
            self.Xmidpoint = ((gameState.getWalls().asList()[-1][0] + 1) / 2)
            self.team = "Blue"
        self.Ydist = gameState.getWalls().asList()[-1][1]
        self.InitialCapsuleList = self.getCapsules(gameState)
        self.InitialFoodList = self.getFood(gameState).asList()
        self.successorScoreWeight = 1000
        self.multiplier = 1.005
        self.nearestFoodWeight = float(self.successorScoreWeight) / (49 + self.multiplier)
        self.nearestCapsuleWeight = self.nearestFoodWeight * self.multiplier
        self.opponentIndices = self.getOpponents(gameState)
        self.totalTime = gameState.data.timeleft

    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest Q(s,a).
    """
        actions = gameState.getLegalActions(self.index)
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        bestActions = random.choice(bestActions)
        return bestActions

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
        # position of our offensive agent after taking the given action
        pos = successor.getAgentPosition(self.index)

        # ExactOpponentGhostDist feature...If we know the exact distance of either or both
        # opponents, then the feature is the distance to the nearest opponent
        distances = []
        features['ExactOpponentGhostDist'] = 0
        features['OnTop'] = 0
        for opp in self.opponentIndices:
            observation = gameState.getAgentPosition(opp)
            if observation is not None:
                if self.team == "Red":
                    if observation[0] > self.Xmidpoint:
                        distances.append(self.getMazeDistance(observation, pos))
                        features['ExactOpponentGhostDist'] = min(distances)
                else:
                    if observation[0] < self.Xmidpoint:
                        distances.append(self.getMazeDistance(observation, pos))
                        features['ExactOpponentGhostDist'] = min(distances)
            if pos == observation:
                features['OnTop'] = 1

        # distToNearestHome feature...the feature is the distance to the nearest safe zone (home)
        distances = []
        for i in range(1, self.Ydist - 1):
            if not gameState.hasWall(self.Xmidpoint, i):
                distances.append((self.getMazeDistance((self.Xmidpoint, i), pos), action))
        features['distToNearestHome'] = min(distances, key=lambda item: item[0])[0]

        # distToNearestFood feature... the feature is the distance to the nearest food
        # distToNearestFood2 feature... the feature is the distance to the 2nd nearest food
        features['distToNearestFood'] = 0
        features['distToNearestFood2'] = 0
        if len(foodList) > 0:
            sort = sorted([self.getMazeDistance(pos, food) for food in foodList])
            minDistanceFood = sort[0]
            if len(sort) > 1:
                minDistanceFood2 = sort[1]
                if minDistanceFood2 != minDistanceFood:
                    features['distToNearestFood2'] = minDistanceFood2
            features['distToNearestFood'] = minDistanceFood

        # distToNearestCapsule feature... the feature is the distance to the nearest capsule
        features['distToNearestCapsule'] = 0
        capsuleDist = []
        if len(capsuleList) > 0:
            for capsule in capsuleList:
                capsuleDist.append(self.getMazeDistance(pos, capsule))
                minDistanceCapsule = min(capsuleDist)
            features['distToNearestCapsule'] = minDistanceCapsule

        # numFoodCarry feature... this feature is the number of food that our pacman is carrying
        numCarrying = successor.getAgentState(self.index).numCarrying
        features['numFoodCarry'] = numCarrying

        # successorScore feature... this makes sure that our agent eats the food/capsule with next action
        # even though the nearestFood/nearestCapsule feature can spike when evaluating its next action
        num = self.algo(50, self.successorScoreWeight, self.nearestFoodWeight)
        features['successorScore'] = - len(foodList) - (num * len(capsuleList))  # self.getScore(successor)

        # ExpectedAgentDist feature... our agent gets some sonar reading as to how close the opponents are.
        # this feature is the minumum of that sonar reading distance
        agentDists = successor.getAgentDistances()
        av1 = (float(agentDists[0]) + agentDists[1]) / 2
        av2 = (float(agentDists[1]) + agentDists[2]) / 2
        features['ExpectedAgentDist'] = min(av1, av2)

        # if our offensive agent is on our half and we see the opponent on our half, then
        # play defense. or our team is winning by a certain amount
        agentPacman = gameState.getAgentState(self.index).isPacman
        oppExactLocations = self.getOpponentExactLocationO(gameState)
        if not agentPacman or self.getScore(gameState) > 3/5 * self.InitialCapsuleList:
            if oppExactLocations:
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

        # print action, features

        return features

    def getWeights(self, gameState, action):
        agentPacman = gameState.getAgentState(self.index).isPacman
        foodList = self.getFood(gameState).asList()
        opponentAttr = self.getOpponentsAttrToOffense(gameState, action)
        numGhostsClose = opponentAttr[0]
        closeGhostsTimerList = opponentAttr[1]
        farGhostsTimerList = opponentAttr[2]
        oppExactLocationsO = self.getOpponentExactLocationO(gameState)
        oppExactLocations = self.getOpponentExactLocation(gameState)
        numCarrying = gameState.getAgentState(self.index).numCarrying
        timeLeft = gameState.data.timeleft
        # number of food that the opponent is carrying
        numFoodOppCarry = []
        for opp in self.opponentIndices:
            numFoodOppCarry.append(gameState.getAgentState(opp).numCarrying)
        # if there are two or less food left
        if len(foodList) <= 2 or \
                (numCarrying > 1 and self.getFeatures(gameState, action)['distToNearestHome'] <= 2) or \
                (numCarrying > len(self.InitialFoodList)/2) or \
                (numCarrying > 7 and self.getFeatures(gameState, action)['distToNearestHome'] <= 8) or \
                (self.getFeatures(gameState, action)['distToNearestHome'] > timeLeft + 100):
            # get back home as quick as possible while staying away from nearest ghost
            return {'successorScore': 0, 'distToNearestFood': 0, 'distToNearestFood2': 0,
                    'distToNearestCapsule': 0, 'ExactOpponentGhostDist': 10, 'distToNearestHome': -20,
                    'numFoodCarry': 0, 'ExpectedAgentDist': 0, 'OnTop': 0}
        # if more than two food left
        else:
            # if our agent is on other side (he is pacman)
            if agentPacman:
                # if at least one opponent is close
                if numGhostsClose > 0:
                    # if close opponent is scared and has more than 7 seconds left on timer
                    if list(closeGhostsTimerList.values()) and max(list(closeGhostsTimerList.values())) > 7:
                        # run away
                        return {'successorScore': 0,
                                'distToNearestFood': 0,
                                'distToNearestFood2': 0,
                                'distToNearestCapsule': 0,
                                'ExactOpponentGhostDist': -200, 'distToNearestHome': 0, 'numFoodCarry': 0,
                                'ExpectedAgentDist': 0,
                                'OnTop': 0}
                    # if close opponent is not scared
                    else:
                        # stay away from close opponent and try to get home
                        return {'successorScore': 0,
                                'distToNearestFood': 0,
                                'distToNearestFood2': 0,
                                'distToNearestCapsule': 0,
                                'ExactOpponentGhostDist': 100, 'distToNearestHome': -2000, 'numFoodCarry': 0,
                                'ExpectedAgentDist': 0,
                                'OnTop': 0}
                # if no close opponents
                else:
                    # if at least one opponent is scared and next action is to kill opponent.
                    # this takes care of the weird case where ExactOpponentGhostDist changes from
                    # a positive value to zero because of eating the scared ghost
                    if list(closeGhostsTimerList.values()) and max(list(closeGhostsTimerList.values())) > 2 and self.getFeatures(gameState, action)['OnTop'] != 0:
                        # then kill opponent
                        return {'successorScore': 0,
                                'distToNearestFood': 0,
                                'distToNearestFood2': 0,
                                'distToNearestCapsule': 0,
                                'ExactOpponentGhostDist': 0, 'distToNearestHome': 0, 'numFoodCarry': 0,
                                'ExpectedAgentDist': 0,
                                'OnTop': 10000}
                    # if at least one opponent is scared
                    elif list(closeGhostsTimerList.values()) and max(list(closeGhostsTimerList.values())) > 2:
                        # don't go after capsules - NOTE: these values don't work as they should right now
                        return {'successorScore': self.successorScoreWeight,
                                'distToNearestFood': -self.nearestFoodWeight,
                                'distToNearestFood2': 0,
                                'distToNearestCapsule': 10,
                                'ExactOpponentGhostDist': 0, 'distToNearestHome': 0, 'numFoodCarry': 0,
                                'ExpectedAgentDist': 0,
                                'OnTop': 0}
                    # if no opponents are scared
                    else:
                        # then act optimally
                        return {'successorScore': self.successorScoreWeight,
                                'distToNearestFood': -self.nearestFoodWeight,
                                'distToNearestFood2': 0,
                                'distToNearestCapsule': -self.nearestCapsuleWeight,
                                'ExactOpponentGhostDist': 0, 'distToNearestHome': 0, 'numFoodCarry': 0,
                                'ExpectedAgentDist': 5,
                                'OnTop': 0}
            # if our agent is on its own side (he is ghost)
            else:
                # if we know position of opponent offensive pacman
                if oppExactLocationsO:
                    agent = DefensiveReflexAgent(gameState)
                    return agent.getWeights(gameState, action)
                # if we don't know the position of opponent offensive guy
                else:
                    # if we know the position of at least one of the opponents
                    if oppExactLocations:
                        return {'successorScore': self.successorScoreWeight,
                                'distToNearestFood': -self.nearestFoodWeight,
                                'distToNearestFood2': 0,
                                'distToNearestCapsule': -self.nearestCapsuleWeight,
                                'ExactOpponentGhostDist': 25, 'distToNearestHome': 0, 'numFoodCarry': 0,
                                'ExpectedAgentDist': 5,
                                'OnTop': 0}
                    else:
                        # then act optimally
                        return {'successorScore': self.successorScoreWeight,
                                'distToNearestFood': -self.nearestFoodWeight,
                                'distToNearestFood2': 0,
                                'distToNearestCapsule': -self.nearestCapsuleWeight,
                                'ExactOpponentGhostDist': 0, 'distToNearestHome': 0, 'numFoodCarry': 0,
                                'ExpectedAgentDist': 5,
                                'OnTop': 0}

    # algorithm that returns a value to be used above that
    # prevents our agent from not consuming food/capsules during the weird case
    def algo(self, distToNearestFood, successorScoreWeight, NearestFoodWeight):
        x = (distToNearestFood - 1) * successorScoreWeight
        y = (distToNearestFood - 1) * (distToNearestFood - 1)
        z = NearestFoodWeight
        c = (x - (y * z) + z) / float(successorScoreWeight)
        return c

    # returns number of close opponents to offensive agent,
    # scaredTimer of close opponents to offensive agent
    # scaredTimer of far opponents to offensive agent
    def getOpponentsAttrToOffense(self, gameState, action):
        closeTimerList = util.Counter()
        notCloseTimerList = util.Counter()
        see_me = 0
        # offensive agent position
        offensiveAgentPos = gameState.getAgentState(self.index).getPosition()
        for opp in self.opponentIndices:
            oppAgentPos = gameState.getAgentState(opp).getPosition()
            # if we know the opponent agent position
            if oppAgentPos is not None:
                dis = self.getMazeDistance(offensiveAgentPos, oppAgentPos)
                # if the opponent agent position is greater than 5 units from our offensive guy
                # then the opponent is not close so we won't worry about it
                if dis > 5:
                    notCloseTimerList[opp] = gameState.getAgentState(opp).scaredTimer
                # if the opponent agent position is less than 5 units from our offensive guy
                # then the opponent is close so we add it to this list to use later
                else:
                    # keeps track of how many close ghosts there are
                    see_me += 1
                    closeTimerList[opp] = gameState.getAgentState(opp).scaredTimer
            else:
                notCloseTimerList[opp] = gameState.getAgentState(opp).scaredTimer

        return see_me, closeTimerList, notCloseTimerList

    # returns a list of opponent positions if the position is known
    def getOpponentExactLocation(self, gameState):
        oppAgentPos = util.Counter()
        for opp in self.opponentIndices:
            knownPos = gameState.getAgentState(opp).getPosition()
            if knownPos is not None:
                oppAgentPos[opp] = knownPos
        return oppAgentPos

    # returns a list of only opposing agents positions that are on our half (they are pacmans)
    # if their position is known
    def getOpponentExactLocationO(self, gameState):
        oppExactLocations = self.getOpponentExactLocation(gameState)
        oppAgentPos = []
        if list(oppExactLocations.values()):
            for x in list(oppExactLocations.values()):
                if self.team == "Red":
                    if x[0] < self.Xmidpoint:
                        oppAgentPos.append(x)
                else:
                    if x[0] > self.Xmidpoint:
                        oppAgentPos.append(x)
        return oppAgentPos


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

        # The number of food we have to defend in the current state and the previous state.
        currNumDefFood = len(self.getFoodYouAreDefending(gameState).asList())
        # print "successor state food", nextNumDefFood, "this state food", thisNumDefFood
        # Accounting for the first few previous states being none
        prevNumDefFood = None
        prev = self.getPreviousObservation()
        if self.time <= 2:
            self.time = self.time + 1
        if self.time > 2:
            prevNumDefFood = len(self.getFoodYouAreDefending(prev).asList())
            self.time = self.time + 1
        # print "this state food", currNumDefFood, "prev state food", prevNumDefFood
        # new feature that isn't used
        features['numFoodLeft'] = currNumDefFood
        #Finds the food that was last eaten as well as the food closest to the food that was just eaten
        eatenFood = None
        closestFoodtoEatenFood = None
        # all the food that we have to defend
        currDefFood = self.getFoodYouAreDefending(gameState).asList()
        # if something was eaten
        if prevNumDefFood > currNumDefFood:
            # find what was eaten
            prevDefFood = self.getFoodYouAreDefending(prev).asList()
            eatenFood = list(set(prevDefFood) - set(currDefFood))[0]
            self.timeFlag = self.time
        # the distances to all foods at a certain time. We find the average of this to hopefully find where food is densest
        distancestoAllFoods = []
        for x in currDefFood:
            distancestoAllFoods.append(self.getMazeDistance(myPos, x))
        averageDist = sum(distancestoAllFoods)/len(distancestoAllFoods)
        features['avgDistFood'] = averageDist
        # print averageDist

        # added a noisy distance feature as well. copy pasted from offensive.
        agentDists = successor.getAgentDistances()
        av1 = (float(agentDists[0]) + agentDists[1]) / 2
        av2 = (float(agentDists[1]) + agentDists[2]) / 2
        features['ExpectedAgentDist'] = min(av1, av2)

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
            # print min(dists)
            # print min(dists)
            # print prevDefFood, "AAAAAAA" ,currDefFood
        # print eatenFood
        # print distancestoAllFoods
        # if something was eaten
        if eatenFood is not None:
            #find the food closest to the food that was eaten
            dist = {}
            for x in currDefFood:
                dist[x] = self.getMazeDistance(eatenFood, x)
            closestFoodtoEatenFood = min(dist, key=dist.get)
            self.savedFood = closestFoodtoEatenFood
            features['eatenFoodPos'] = self.getMazeDistance(myPos, closestFoodtoEatenFood)
        else:
            features['eatenFoodPos'] = 0
        if self.timeFlag != 0:
            # print self.time - self.timeFlag
            if self.time - self.timeFlag < 20:
                # print self.savedFood
                # add a thing where if we can see set to 0 instead
                if features['invaderDistance'] == 0:
                    features['eatenFoodPos'] = self.getMazeDistance(myPos, self.savedFood)
                else:
                    features['eatenFoodPos'] = 0
        # print "Eaten Food", eatenFood, "Closest Food to Eaten Food", closestFoodtoEatenFood
        # print numDefFood
        # These make sure pacman doesn't get stuck anywhere.
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -15, 'stop': -100, 'reverse': -2,
                'ExpectedAgentDist': -5, 'numFoodLeft': 0, 'eatenFoodPos': -10, 'avgDistFood': -5}
