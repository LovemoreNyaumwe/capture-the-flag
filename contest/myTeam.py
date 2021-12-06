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
        # print "------main------"
        # print self.opponentIndices

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
        # print self.getMazeDistance(successor.getAgentPosition(self.index), successor.getAgentPosition(2)), self.getMazeDistance(successor.getAgentPosition(self.index), successor.getAgentPosition(4))
        av1 = (float(agentDists[0]) + agentDists[1]) / 2
        av2 = (float(agentDists[2]) + agentDists[3]) / 2
        features['ExpectedAgentDist'] = min(av1, av2)

        # if our offensive agent is on our half and we see the opponent on our half, then
        # play defense. or our team is winning by a certain amount
        agentPacman = gameState.getAgentState(self.index).isPacman
        oppExactLocations = self.getOpponentExactLocationO(gameState)
        if not agentPacman:
            if oppExactLocations:
                features = util.Counter()
                successor = self.getSuccessor(gameState, action)
                myState = successor.getAgentState(self.index)
                myPos = myState.getPosition()

                # particleFilter = ParticleFilter()
                # particleFilter.registerInitialState(gameState, self)
                #
                # particleFilter.observe(self.opponentIndices[0], self.index, gameState)
                # particleFilter.observe(self.opponentIndices[1], self.index, gameState)
                #
                # particleFilter.elapseTime(self.opponentIndices[0], gameState)
                # particleFilter.elapseTime(self.opponentIndices[0], gameState)

                # print self.opponentIndices[0]
                # print self.opponentIndices[1]
                # opponents = self.getOpponents(gameState)
                # print opponents

                # print particleFilter.getBestPositionEstimate(self.opponentIndices[0]), gameState.getAgentState(opponents[0]).getPosition()
                # print "first"
                # print particleFilter.getBestPositionEstimate(self.opponentIndices[1]), gameState.getAgentState(opponents[0]).getPosition()
                # print "second"
                # self.displayDistributionsOverPositions([particleFilter.getBeliefDistribution(self.opponentIndices[0]), particleFilter.getBeliefDistribution(self.opponentIndices[1])])

                # The number of food we have to defend in the current state and the previous state.
                currNumDefFood = len(self.getFoodYouAreDefending(gameState).asList())
                # print "successor state food", nextNumDefFood, "this state food", thisNumDefFood
                # Accounting for the first few previous states being none
                prevNumDefFood = None
                prev = self.getPreviousObservation()
                if self.time <= 2:
                    self.time = self.time + 1
                if self.time > 2 and prev is not None:
                    prevNumDefFood = len(self.getFoodYouAreDefending(prev).asList())
                    self.time = self.time + 1
                else:
                    self.time = self.time + 1
                # print "this state food", currNumDefFood, "prev state food", prevNumDefFood
                # new feature that isn't used
                features['numFoodLeft'] = currNumDefFood
                # Finds the food that was last eaten as well as the food closest to the food that was just eaten
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
                averageDist = sum(distancestoAllFoods) / len(distancestoAllFoods)
                features['avgDistFood'] = averageDist
                # print averageDist

                # added a noisy distance feature as well. copy pasted from offensive.
                agentDists = successor.getAgentDistances()
                av1 = (float(agentDists[0]) + agentDists[1]) / 2
                av2 = (float(agentDists[2]) + agentDists[3]) / 2
                features['ExpectedAgentDist'] = min(av1, av2)
                # pos1 = particleFilter.getBestPositionEstimate(self.opponentIndices[0])
                # pos2 = particleFilter.getBestPositionEstimate(self.opponentIndices[1])
                # if self.team == "Red":
                #     if pos1[0] < self.Xmidpoint:
                #         # print pos1, pos2, "HI, INVADER"
                #         print self.team
                # else:
                #     # print "OUTER", pos1[0], pos2[0], self.Xmidpoint
                #     if pos2[0] > self.Xmidpoint or pos1[0] > self.Xmidpoint:
                #         # print self.team
                #         # print "INNER", pos1[0], pos2[0], self.Xmidpoint
                #         print pos1, pos2, "HI, INVADER"
                # minEst = min(self.getMazeDistance(myPos, pos1),
                #              self.getMazeDistance(myPos, pos2))
                # print minEst
                # features['ExpectedAgentDist'] = minEst

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
                    # find the food closest to the food that was eaten
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
                (numCarrying > len(self.InitialFoodList) / 2) or \
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
                    # if close opponent is scared and has more than 10 seconds left on timer
                    if list(closeGhostsTimerList.values()) and max(list(closeGhostsTimerList.values())) > 10:
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
                                'distToNearestFood': 10,
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
                    if list(closeGhostsTimerList.values()) and max(list(closeGhostsTimerList.values())) > 2 and \
                            self.getFeatures(gameState, action)['OnTop'] != 0:
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
                                'distToNearestFood': -100*self.nearestFoodWeight,
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
                                'ExactOpponentGhostDist': 100, 'distToNearestHome': 0, 'numFoodCarry': 0,
                                'ExpectedAgentDist': 0,
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

        # particleFilter = ParticleFilter()
        # particleFilter.registerInitialState(gameState, self)
        #
        # particleFilter.observe(self.opponentIndices[0], self.index, gameState)
        # particleFilter.observe(self.opponentIndices[1], self.index, gameState)
        #
        # particleFilter.elapseTime(self.opponentIndices[0], gameState)
        # particleFilter.elapseTime(self.opponentIndices[0], gameState)

        # print self.opponentIndices[0]
        # print self.opponentIndices[1]
        #opponents = self.getOpponents(gameState)
        # print opponents

        # print particleFilter.getBestPositionEstimate(self.opponentIndices[0]), gameState.getAgentState(opponents[0]).getPosition()
        # print "first"
        # print particleFilter.getBestPositionEstimate(self.opponentIndices[1]), gameState.getAgentState(opponents[0]).getPosition()
        # print "second"
        # self.displayDistributionsOverPositions([particleFilter.getBeliefDistribution(self.opponentIndices[0]), particleFilter.getBeliefDistribution(self.opponentIndices[1])])

        # The number of food we have to defend in the current state and the previous state.
        currNumDefFood = len(self.getFoodYouAreDefending(gameState).asList())
        # print "successor state food", nextNumDefFood, "this state food", thisNumDefFood
        # Accounting for the first few previous states being none
        prevNumDefFood = None
        prev = self.getPreviousObservation()
        if self.time <= 2:
            self.time = self.time + 1
        if self.time > 2 and prev is not None:
            prevNumDefFood = len(self.getFoodYouAreDefending(prev).asList())
            self.time = self.time + 1
        else:
            self.time = self.time +1
        # print "this state food", currNumDefFood, "prev state food", prevNumDefFood
        # new feature that isn't used
        features['numFoodLeft'] = currNumDefFood
        # Finds the food that was last eaten as well as the food closest to the food that was just eaten
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
                averageDist = sum(distancestoAllFoods) / len(distancestoAllFoods)
        features['avgDistFood'] = averageDist
        # print averageDist

        # added a noisy distance feature as well. copy pasted from offensive.
        agentDists = successor.getAgentDistances()
        av1 = (float(agentDists[0]) + agentDists[1]) / 2
        av2 = (float(agentDists[2]) + agentDists[3]) / 2
        features['ExpectedAgentDist'] = min(av1, av2)
        # pos1 = particleFilter.getBestPositionEstimate(self.opponentIndices[0])
        # pos2 = particleFilter.getBestPositionEstimate(self.opponentIndices[1])
        # if self.team == "Red":
        #     if pos1[0] < self.Xmidpoint:
        #         # print pos1, pos2, "HI, INVADER"
        #         print self.team
        # else:
        #     # print "OUTER", pos1[0], pos2[0], self.Xmidpoint
        #     if pos2[0] > self.Xmidpoint or pos1[0] > self.Xmidpoint:
        #         # print self.team
        #         # print "INNER", pos1[0], pos2[0], self.Xmidpoint
        #         print pos1, pos2, "HI, INVADER"
        # minEst = min(self.getMazeDistance(myPos, pos1),
        #              self.getMazeDistance(myPos, pos2))
        # print minEst
        # features['ExpectedAgentDist'] = minEst

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
            # find the food closest to the food that was eaten
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
        # scaredTimers = []
        # for x in self.getTeam(gameState):
        #     # print self.team
        #     if self.team == "Red":
        #         if gameState.getAgentPosition(x)[0] < self.Xmidpoint:
        #             scaredTimers.append(gameState.getAgentState(x).scaredTimer)
        #     else:
        #         if gameState.getAgentPosition(x)[0] > self.Xmidpoint:
        #             scaredTimers.append(gameState.getAgentState(x).scaredTimer)
        scaredTimers =[]
        for x in self.getTeam(gameState):
            scaredTimers.append(gameState.getAgentState(x).scaredTimer)
        # print scaredTimers
        if max(scaredTimers) > 0:
            return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 15, 'stop': -100, 'reverse': -2,
                'ExpectedAgentDist': -5, 'numFoodLeft': 0, 'eatenFoodPos': -10, 'avgDistFood': -5}
        else:
            return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -15, 'stop': -100, 'reverse': -2,
                    'ExpectedAgentDist': -5, 'numFoodLeft': 0, 'eatenFoodPos': -10, 'avgDistFood': -5}

#
# class ParticleFilter:
#     """
#     A particle filter for approximately tracking a single ghost.
#     Useful helper functions will include random.choice, which chooses an element
#     from a list uniformly at random, and util.sample, which samples a key from a
#     Counter by treating its values as probabilities.
#     """
#
#     def __init__(self, numParticles=300):
#         "Sets the ghost agent for later access"
#         # ReflexCaptureAgent.registerInitialState(self, gameState)
#         # self.index = index
#         # self.players = self.getTeam(gameState)
#         # self.obs = []  # most recent observation position
#         self.setNumParticles(numParticles)
#         # self.beliefDistribution = {}
#         # self.opponents =
#         # self.team = gameState.getTeam()
#         # self.obs = []  # most recent observation position
#         # self.setNumParticles(numParticles)
#
#     # def initializeBelief(self, enemy):
#     #     allPossible = util.Counter()
#     #     for pos in self.particles:
#     #         allPossible[pos] += 1
#     #     allPossible.normalize()
#     #     self.beliefDistribution[enemy] = allPossible
#     def registerInitialState(self, gameState, agent):
#         self.legalPositions = gameState.getWalls().asList(False)
#         # print self.legalPositions
#         self.opponentIndices = agent.getOpponents(gameState)
#         # print "------ris------"
#         # print self.opponentIndices
#         self.initializeUniformly()
#
#     def setNumParticles(self, numParticles):
#         self.numParticles = numParticles
#
#     def getPositionDistribution(self, enemy):
#         """
#         Returns a distribution over successor positions of the ghost from the
#         given gameState.
#         You must first place the ghost in the gameState, using setGhostPosition
#         below.
#         """
#         # print enemy
#         x, y = enemy  # The position you set
#         # print ghostPosition
#         # actionDist = self.ghostAgent.getDistribution(gameState)
#         # the ghost cannot fly so the possible ghost positions are up, down, left and right
#         up = (x, y + 1)
#         down = (x, y - 1)
#         left = (x - 1, y)
#         right = (x + 1, y)
#         positions = [up, down, left, right]
#         # print positions
#         # print self.legalPositions
#         possiblePositions = list(set(positions) & set(self.legalPositions))
#         # print possiblePositions
#
#         # the actions have equal probability since the agent can take either of them
#         actionDist = util.Counter()
#         for pos in possiblePositions:
#             actionDist[pos] += 1
#         actionDist.normalize()
#         return actionDist
#
#         # dist = util.Counter()
#         # for action, prob in actionDist.items():
#         #     successorPosition = game.Actions.getSuccessor(ghostPosition, action)
#         #     dist[successorPosition] = prob
#         # return dist
#
#     # def setGhostPosition(self, gameState, ghostPosition):
#     #     """
#     #     Sets the position of the ghost for this inference module to the
#     #     specified position in the supplied gameState.
#     #     Note that calling setGhostPosition does not change the position of the
#     #     ghost in the GameState object used for tracking the true progression of
#     #     the game.  The code in inference.py only ever receives a deep copy of
#     #     the GameState object which is responsible for maintaining game state,
#     #     not a reference to the original object.  Note also that the ghost
#     #     distance observations are stored at the time the GameState object is
#     #     created, so changing the position of the ghost will not affect the
#     #     functioning of observeState.
#     #     """
#     #     conf = game.Configuration(ghostPosition, game.Directions.STOP)
#     #     gameState.data.agentStates[self.index] = game.AgentState(conf, False)
#     #     return gameState
#
#     def initializeUniformly(self, enemy=None):
#         """
#         Initializes a list of particles. Use self.numParticles for the number of
#         particles. Use self.legalPositions for the legal board positions where a
#         particle could be located.  Particles should be evenly (not randomly)
#         distributed across positions in order to ensure a uniform prior.
#         Note: the variable you store your particles in must be a list; a list is
#         simply a collection of unweighted variables (positions in this case).
#         Storing your particles as a Counter (where there could be an associated
#         weight with each position) is incorrect and may produce errors.
#         """
#         "*** YOUR CODE HERE ***"
#         # print self.numParticles
#         # print self.legalPositions
#         # making an empty list that will hold all my particles
#         # self.setNumParticles(10000)
#         # print self.numParticles
#         self.particles = []
#         # making a counter that will show how many particles I still have to allocate
#         particlesLeft = self.numParticles
#         # while there are still particles, we'll distribute them over the different positions
#         # print self.legalPositions
#         while particlesLeft != 0:
#             # splitting up by position
#             for x in self.legalPositions:
#                 # checking again if there are still particles left, as we may have run out before hitting the while again
#                 if particlesLeft != 0:
#                     # add that position to the particles list
#                     self.particles.append(x)
#                     # we've allocated a particle, so subtract it from how many left.
#                     particlesLeft -= 1
#         # partition the particles into enemy one particles and enemy two particles or both at the beginning
#         self.particlesEnemy0 = []
#         self.particlesEnemy1 = []
#         if enemy == self.opponentIndices[0]:
#             self.particlesEnemy0 = self.particles
#         if enemy == self.opponentIndices[1]:
#             self.particlesEnemy1 = self.particles
#         if enemy is None:
#             self.particlesEnemy0 = self.particles
#             self.particlesEnemy1 = self.particles
#
#         # if enemy == self.opponentIndices[0]
#
#         # print self.particles
#         # print len(self.particles)
#
#     def observe(self, enemy, pacman, gameState):
#         """
#         Update beliefs based on the given distance observation. Make sure to
#         handle the special case where all particles have weight 0 after
#         reweighting based on observation. If this happens, resample particles
#         uniformly at random from the set of legal positions
#         (self.legalPositions).
#         A correct implementation will handle two special cases:
#           1) When a ghost is captured by Pacman, all particles should be updated
#              so that the ghost appears in its prison cell,
#              self.getJailPosition()
#              As before, you can check if a ghost has been captured by Pacman by
#              checking if it has a noisyDistance of None.
#           2) When all particles receive 0 weight, they should be recreated from
#              the prior distribution by calling initializeUniformly. The total
#              weight for a belief distribution can be found by calling totalCount
#              on a Counter object
#         util.sample(Counter object) is a helper method to generate a sample from
#         a belief distribution.
#         You may also want to use util.manhattanDistance to calculate the
#         distance between a particle and Pacman's position.
#         """
#         # print enemy
#         # exit(1)
#         noisyDistances = gameState.getAgentDistances()
#         noisyDistance = noisyDistances[enemy]
#         pacmanPosition = gameState.getAgentPosition(pacman)
#         emissionModel = self.getBeliefDistribution(enemy)
#         # print emissionModel
#         "*** YOUR CODE HERE ***"
#         allPossible = util.Counter()
#         if noisyDistance is not None:
#             for x in self.particles:
#                 trueDistance = util.manhattanDistance(x, pacmanPosition)
#                 prob = gameState.getDistanceProb(trueDistance, noisyDistance)
#                 allPossible[x] = emissionModel[x] * prob
#
#             # check if all 0
#             if allPossible.totalCount() == 0:
#                 self.initializeUniformly(gameState)
#             else:
#                 # resample
#                 newParticles = []
#                 for i in range(self.numParticles):
#                     newParticles.append(util.sample(allPossible))
#                 # update the new particles based on the likelihood of the different positions
#                 if enemy == self.opponentIndices[0]:
#                     self.particlesEnemy0 = newParticles
#                 if enemy == self.opponentIndices[1]:
#                     self.particlesEnemy1 = newParticles
#
#     def elapseTime(self, enemy, gameState):
#         """
#         Update beliefs for a time step elapsing.
#         As in the elapseTime method of ExactInference, you should use:
#           newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
#         to obtain the distribution over new positions for the ghost, given its
#         previous position (oldPos) as well as Pacman's current position.
#         util.sample(Counter object) is a helper method to generate a sample from
#         a belief distribution.
#         """
#         "*** YOUR CODE HERE ***"
#         # print enemy
#         # print self.opponentIndices[0], self.opponentIndices[1]
#         if enemy == self.opponentIndices[0]:
#             # i basically changed what was done earlier to account for particles in a list instead of a counter.
#             # create a new list
#             particles = []
#             # loop over the list of particles (basically valid positions)
#             # print self.particles
#             for oldPos in self.particlesEnemy0:
#                 # same as before
#                 newPosDist = self.getPositionDistribution(oldPos)
#                 # we generate the sample from the pos dist found and append the result to the new list
#                 particles.append(util.sample(newPosDist))
#             # we set the particle list to the new sampled one, updating it!
#             # update the new particles based on the likelihood of the different positions
#             self.particlesEnemy0 = particles
#         if enemy == self.opponentIndices[1]:
#             # i basically changed what was done earlier to account for particles in a list instead of a counter.
#             # create a new list
#             particles = []
#             # loop over the list of particles (basically valid positions)
#             # print self.particles
#             for oldPos in self.particlesEnemy1:
#                 # same as before
#                 newPosDist = self.getPositionDistribution(oldPos)
#                 # we generate the sample from the pos dist found and append the result to the new list
#                 particles.append(util.sample(newPosDist))
#             # we set the particle list to the new sampled one, updating it!
#             # update the new particles based on the likelihood of the different positions
#             self.particlesEnemy1 = particles
#
#     def getBeliefDistribution(self, enemy):
#         """
#         Return the agent's current belief state, a distribution over ghost
#         locations conditioned on all evidence and time passage. This method
#         essentially converts a list of particles into a belief distribution (a
#         Counter object)
#         """
#         "*** YOUR CODE HERE ***"
#         count = util.Counter()
#         if enemy == self.opponentIndices[0]:
#             for particle in self.particlesEnemy0:
#                 count[particle] += 1
#             count.normalize()
#         if enemy == self.opponentIndices[1]:
#             for particle in self.particlesEnemy1:
#                 count[particle] += 1
#             count.normalize()
#         return count
#
#     def getBestPositionEstimate(self, enemy):
#         belief = self.getBeliefDistribution(enemy)
#         return belief.argMax()
