# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
import sys
from game import Directions, Actions
import game
from util import nearestPoint
from distanceCalculator import manhattanDistance
sys.path.append('teams/AIGOGOGO/')

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='Attacker', second='prophet'):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).
        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
        IMPORTANT: This method may run for at most 15 seconds.
        """



        CaptureAgent.registerInitialState(self, gameState)
        self.home = gameState.getAgentState(self.index).getPosition()
        self.defendFood=len(self.getFoodYouAreDefending(gameState).asList())
        self.lastEaten=None
        self.eatenFood=None
        self.walls = gameState.getWalls()
        self.getmycap = self.getCapsulesYouAreDefending(gameState)
        self.initialPosition = gameState.getInitialAgentPosition(self.index)
        self.ghostPos = None
        self.backhome = False
        self.checkScare = False
        self.middle = []
        self.foodList = len(self.getFood(gameState).asList())
        # Initialization calculate the homepoint
        # homepoint means the points along the middle Line of layout
        # Red on left, Blue on right, therefore blue needs + 1
        # pacman go to homepoint to score and turns into ghost

        
        if self.red:
            central = (gameState.data.layout.width - 2) / 2 - 1
            self.ghostPos = (gameState.data.layout.width - 2,gameState.data.layout.height - 2)
        else:
            central = ((gameState.data.layout.width -2) / 2)
            self.ghostPos = (1,1)

        for height in range(0, gameState.data.layout.height):
            if not (central, height) in gameState.getWalls().asList():
                self.middle.append((central, height))
    def getSuccessor(self, gameState, action):

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights
    
    
    def getDefenders(self,gameState):
        enemies=[]
        defenders=[]
        for i in self.getOpponents(gameState):
            enemies.append(gameState.getAgentState(i))
        for e in enemies:
            if e.getPosition() != None and not e.isPacman:
                defenders.append(e)
        if len(defenders)==0:
          return  None
        else:
          return defenders
    def getInvaders(self, gameState):
        enemies=[]
        invaders=[]
        for i in self.getOpponents(gameState):
            enemies.append(gameState.getAgentState(i))
        for e in enemies:
            if e.getPosition() != None and e.isPacman:
                invaders.append(e)
        if len(invaders)==0:
          return None
        else:
          return invaders
    def waStarSearch(self,gameState,goal,heuristic):
        w = 2
        action_list = []
        visted = []
        cost = 0
        start =self.getCurrentObservation().getAgentState(self.index).getPosition()
        priorityQueue = util.PriorityQueue()
        priorityQueue.push((start,action_list),cost)
        while not priorityQueue.isEmpty():
            current_node, action = priorityQueue.pop()
            if goal == current_node:
              if len(action) == 0:
                    return 'Stop'
              return action[0]
            if current_node not in visted:
                expand = self.getheuSuccessors(current_node)
                visted.append(current_node)
                for arg in expand:
                    location = arg[0]
                    direction = arg[1]
                    if (location not in visted):
                        priorityQueue.push((location, action + [direction]),len(action +[direction]) + w * heuristic(gameState,location))
        return 'Stop'
    def allHeuristic(self,gameState,thisPosition):
            heuristics=[]
            closedis = []
            ghoasts=self.getDefenders(gameState)
            if ghoasts!=None:
              for i in ghoasts:
                closedis.append(self.getMazeDistance(thisPosition,i.getPosition()))
              d=min(closedis)
              if d<=3:
                heuristics.append((5-d)**3)
              else:
                heuristics.append(0)
            else:
              return 0
            return max(heuristics)
    def getMiddleLines(self,gameState):
        middle = []
        line = []
        if self.red:
          for i in range(0, gameState.data.layout.height):
                middle.append(((gameState.data.layout.width / 2) - 1, i))
        else:
          for i in range(0, gameState.data.layout.height):
                middle.append(((gameState.data.layout.width / 2), i))
        for i in middle:
              if i not in gameState.getWalls().asList():
                    line.append(i)
        return line
    def getheuSuccessors(self, currentPosition):
        successors = []
        forbidden =self.walls
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
          x, y = currentPosition
          dx, dy = Actions.directionToVector(action)
          nextx, nexty = int(x + dx), int(y + dy)
          if not self.walls[nextx][nexty]:
            nextPosition = (nextx, nexty)
            successors.append((nextPosition, action))
        return successors
    def simpleHeuristic(self,gameState,thisPosition):
        return 0


class Node:
    """
    A node in a monte_carlo_tree of pacman using in this
    project.
    """
    def __init__(self, gameState, reward, visited, parent=None, childs=None):
        self.gameState = gameState
        self.reward = reward
        self.visited = visited
        self.parent = parent
        self.childs = childs # childs is a list of node

    def addChild(self, c):
        if self.childs is None:
            self.childs = [c]
        else:
            self.childs += [c]
        c.parent = self


        

class Attacker(ReflexCaptureAgent):

        

    def chooseAction(self, gameState):
        # Get remain food as a list
        foodLeft = len(self.getFood(gameState).asList())

        # food our agent carrying in this gameState
        foodNum = gameState.getAgentState(self.index).numCarrying

        # if our agent eat food in last state, set preFoodNum to 1, else 0
        # if self.getPreviousObservation() is not None:
        #     preFoodNum = self.getPreviousObservation().getAgentState(self.index).numCarrying
        # else:
        #     preFoodNum = 0

        # enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        # chaseGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer is 0]
        # if preFoodNum == foodNum and len(chaseGhosts) == 0:
        #     self.noFoodTimer += 1
        # else:
        #     self.noFoodTimer = 0
        
        # Iteration of Monte Carlo Tree
        # Simulation Depth of Monte Carlo

        # Timer use to see the performance
        self.backhome = False
        chaseGhosts = []
        defender = self.getDefenders(gameState)
        if defender != None:
            for i in defender:
                if i.scaredTimer<5:
                    chaseGhosts.append(i)
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        capsuleList = self.getCapsules(gameState)
        clist = []
        closeCapsules = None
        if len(capsuleList) > 0:
            for c in capsuleList:
                clist.append(self.getMazeDistance(c, gameState.getAgentPosition(self.index)))
            close = min(clist)
            capsuleDis = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), c) for c in capsuleList]
            closeCapsules=[c for c,d in zip(self.getCapsules(gameState),capsuleDis) if d==min(capsuleDis)]
        
        if len(chaseGhosts) > 0:
            gDistL = []
            gMDistL = []
            for i in chaseGhosts:
                gDistL.append(self.getMazeDistance(myPos ,i.getPosition()))
            for i in chaseGhosts:
                gMDistL.append(manhattanDistance(myPos, i.getPosition()))
            gDist = min(gDistL)
            gMDist = min(gMDistL)
        if gameState.getAgentState(self.index).numCarrying > 0 and gameState.data.timeleft< 150:
            self.backhome =  True
        elif len(self.getFood(gameState).asList()) <= 2:
            self.backhome = True
            
        elif myState.isPacman and gameState.getAgentState(self.index).numCarrying >= 6 and len(chaseGhosts) > 0:
        	if gMDist <= 5:
        		self.backhome = True
        		# self.survivalPoint = self.start
        elif myState.isPacman and gameState.getAgentState(self.index).numCarrying  >= 1 and len(chaseGhosts)> 0:
            if gDist <= 5:
                self.backhome = True
            # self.survivalPoint = self.start
        # elif myState.isPacman and gameState.getAgentState(self.index).numCarrying  < 1 and len(chaseGhosts)> 0:
        #     self.backhome =  True
        # elif myState.isPacman and gameState.getAgentState(self.index).numCarrying  < 1 and len(chaseGhosts)> 0 and len(capsuleList) <= 0:
        #     print("no move")
        #     if gDist <= 5 or gMDist <= 5:
        #         self.backhome =  True
            # self.survivalPoint = self.start
        # self.powerCheck(gameState)
        scareTime = []
        if defender != None:
            for i in defender:
                scareTime.append(i.scaredTimer)
        if len(scareTime)>0:
            if min(scareTime) >= 15:
                self.checkScare = True
                self.backhome = False
            else:
                self.checkScare = False
        # if not gameState.getAgentState(self.index).isPacman:
        #     self.survivalPoint = self.start

        # Choose an action depending on survivalMode
        # if survivalMode is on, we use go back home directly,
        # else, we use MCT to pick a best action for us
        if self.backhome is False:
            values = []
            MCT = self.MCT(gameState, iter_times=32,simulate_depth=10)

            childs = MCT.childs
            for c in childs:
                values.append(c.reward/c.visited)
            max_value = max(values)
            nextNode = []
            for m,n in zip(childs,values):
                if n == max_value:
                    nextNode.append(m)
            nextState = nextNode[0]
            return nextState.gameState.getAgentState(self.index).configuration.direction
        else:
            action = self.goBackHome(gameState)
            return action

    def getCloseFood(self, gameState):
        foods = []
        for i in self.getFood(gameState).asList():
            foods.append(i)
        
        
        if len(foods)==0:
          return None
        else:
          foodDistance= []
          for i in foods:
              foodDistance.append(self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), i))
          closeFood = min(foodDistance)
          return closeFood

    def goBackHome(self, gameState):
        
        # Check Agent is closer to capsule or Ghost closer to capsule
        # If agent closer, then go and get that capsule
        capsuleList = self.getCapsules(gameState)
        middleLines = self.getMiddleLines(gameState)
        middleDis = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), mi) for mi in
                 middleLines]
        closeMiddle = [m for m, d in zip(middleLines, middleDis) if d == min(middleDis)]
        middle = closeMiddle[0]
        gotoCap = False
        clist = []
        closeCapsules = None
        if len(capsuleList) > 0 and self.getDefenders(gameState) != None:
            for c in capsuleList:
                clist.append(self.getMazeDistance(c, gameState.getAgentPosition(self.index)))
            close = min(clist)
            capsuleDis = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), c) for c in capsuleList]
            closeCapsules=[c for c,d in zip(self.getCapsules(gameState),capsuleDis) if d==min(capsuleDis)]
            if close < min(middleDis):
                gotoCap = True
        if gotoCap and len(closeCapsules) != 0:
            return self.waStarSearch(gameState, closeCapsules[0],self.allHeuristic)
        return self.waStarSearch(gameState, middle,self.allHeuristic)
    def removeReverse(self, gameState):
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        currentDirection = gameState.getAgentState(self.index).configuration.direction
        reversedDirection = Actions.reverseDirection(gameState.getAgentState(self.index).configuration.direction)
        if reversedDirection in actions and len(actions) > 1:
            actions.remove(reversedDirection)
        return actions
    def checkFood(self, gameState, action, depth):
        flag = self.foodDecide(gameState, action, depth)
        if flag:
            return True
        new_state = gameState.generateSuccessor(self.index, action)
        actions = self.removeReverse(new_state)
        if len(actions) == 0:
            return False
        for a in actions:
            if self.checkFood(new_state, a, depth - 1):
                return True
        return False
    
    def foodDecide(self, gameState, action, depth):
        if depth == 0:
            return True
        new_state = gameState.generateSuccessor(self.index, action)
        capList = self.getCapsules(gameState)
        if gameState.getAgentState(self.index).numCarrying < new_state.getAgentState(self.index).numCarrying:
            return True
        if new_state.getAgentPosition(self.index) in capList:
            return True

    # def checkEmptyPath(self, gameState, action, depth):
    #     if depth == 0:
    #         return False
    #     # check whether score changes
    #     successor = gameState.generateSuccessor(self.index, action)
    #     score = gameState.getAgentState(self.index).numCarrying
    #     newScore = successor.getAgentState(self.index).numCarrying

    #     capList = self.getCapsules(gameState)
    #     myPos = successor.getAgentPosition(self.index)

    #     if myPos in capList:
    #     	return False

    #     if score < newScore:
    #         return False

    #     # the action has taken by successor gamestate
    #     actions = successor.getLegalActions(self.index)
    #     actions.remove(Directions.STOP)

    #     curDirct = successor.getAgentState(self.index).configuration.direction
    #     revDirct = Directions.REVERSE[curDirct]

    #     if revDirct in actions:
    #         actions.remove(revDirct)

    #     if len(actions) == 0:
    #         return True
    #     for action in actions:
    #         if not self.checkEmptyPath(successor, action, depth-1):
    #             return False
    #     return True

    def MCT(self, gameState, iter_times, simulate_depth):
        root = Node(gameState, 0.0, 0)
        actions = root.gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        successors = []
        for action in actions:
            if not self.checkFood(root.gameState, action, 8):
                actions.remove(action)
        for action in actions:
            successors.append(self.getSuccessor(root.gameState, action))

        for suc in successors:
            child = Node(suc, 0.0, 0)
            root.addChild(child)

        for i in range(iter_times):
            curNode = root
            childs = curNode.childs
            while childs is not None:
                greedy_max = []
                for c in childs:
                    greedy_max.append(c.reward)
                greedy_max_values = max(greedy_max)
                curNode = None
                nextNode = []
                for m,n in zip(childs, greedy_max):
                    if n == greedy_max_values:
                        nextNode.append(m)
                curNode = nextNode[0]
                childs = curNode.childs
            if curNode.visited != 0:
                actions = curNode.gameState.getLegalActions(self.index)
                actions.remove(Directions.STOP)
                successors = []
                for action in actions:
                    if not self.checkFood(curNode.gameState, action, 8):
                        actions.remove(action)
                for action in actions:
                    successors.append(self.getSuccessor(curNode.gameState, action))
                for suc in successors:
                    sucPos = suc.getAgentPosition(self.index)
                    child = Node(suc, 0.0, 0)
                    curNode.addChild(child)
                curNode = random.choice(curNode.childs)
            reward = self.simulate(curNode.gameState, simulate_depth)
            # while curNode.parent is not None:
            #     curNode.reward += reward
            #     curNode.visited += 1
            #     curNode = curNode.parent 
            # root.reward += reward
            # root.visited += 1
            root = self.backpropogation(root,curNode,reward)
        return root
    

    def backpropogation(self,root,curNode,reward):
        while True:
            if curNode.parent is not None:
                curNode.reward += reward
                curNode.visited += 1
                curNode = curNode.parent
            else:
                break
        root.reward += reward
        root.visited += 1
        return root

    def simulate(self, gameState, level):
        reward = 0
        for i in range(level):
            legalActions = gameState.getLegalActions(self.index)
            legalActions.remove(Directions.STOP)
            nextAction = random.choice(legalActions)

            suc = self.getSuccessor(gameState, nextAction)
            sucPos = suc.getAgentPosition(self.index)
            reward += self.evaluate(gameState, nextAction)
            if not suc.getAgentState(self.index).isPacman:
                if self.getDefenders(suc) != None:
                    dtoGhost = 500
                    for i in self.getDefenders(suc):
                        d = self.getMazeDistance(sucPos, i.getPosition())
                        if d < dtoGhost:
                            dtoGhost = d
                            self.ghostPos = i.getPosition()
            # if not suc.getAgentState(self.index).isPacman:
            #     dtoGhost = 99999.9
            #     opStates = [suc.getAgentState(i) for i in self.getOpponents(suc)]
            #     for op in opStates:
            #         if op.getPosition() is None:
            #             continue
            #         if not op.isPacman:
            #             d = self.getMazeDistance(sucPos, op.getPosition())
            #             if d < dtoGhost:
            #                 dtoGhost = d
            #                 self.ghostPos = op.getPosition()
        return reward

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        foodList = self.getFood(successor).asList()
        capsuleList = self.getCapsules(successor)    
        features['successorScore'] = -len(foodList)
        features['eatenCap'] = -len(capsuleList)
        clist = []
        closeCapsules = None
        if len(capsuleList) > 0:
            for c in capsuleList:
                clist.append(self.getMazeDistance(c, gameState.getAgentPosition(self.index)))
            close = min(clist)
            capsuleDis = [self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), c) for c in capsuleList]
            closeCapsules=[c for c,d in zip(self.getCapsules(gameState),capsuleDis) if d==min(capsuleDis)]

        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        myPos = myState.getPosition()
        if len(foodList) > 0:
            minDistance =self.getCloseFood(successor)
            features['distanceToFood'] = minDistance

        # features['distanceToOp'] = 0
        # op_indeics = self.getOpponents(gameState)
        if self.getDefenders(gameState) != None:
            for i in self.getDefenders(gameState):
                    if self.getMazeDistance(i.getPosition(), myPos) == 1:
                        features['distanceToOp'] = -100
                    # elif self.getMazeDistance(i.getPosition(), myPos) <= 5 and self.getMazeDistance(i.getPosition(), myPos) > 1 and len(capsuleList) > 0:
                    #     features['eatenCap'] = 100
                    else:
                        features['distanceToOp'] = self.getMazeDistance(myPos, i.getPosition())
        else:
            features['distanceToOp'] = 0
        # if self.getDefenders(gameState) != None and len(capsuleList) != 0:
        #     for i in self.getDefenders(gameState):
        #             if self.getMazeDistance(i.getPosition(), myPos) <= 5:
        #                 features['eatenCap'] = self.getMazeDistance(myPos, closeCapsules[0])
                

        # BEST_ENTRY
        features['distanceToEntry'] = 0
        if not gameState.getAgentState(self.index).isPacman:
            final = None
            dAndG = 0.0
            for i in self.middle:
                d = self.getMazeDistance(self.ghostPos,i)
                if d > dAndG:
                    dAndG = d
                    final = i
            features['distanceToEntry'] = self.getMazeDistance(myPos,final)
        # BEST_ENTRY END
        # features['arrivedHome'] = self.getScore(successor)
        
        # features['dead'] = 0
        if successor.getAgentState(self.index).getPosition() is self.initialPosition:
            features['dead'] = 1
        else:
            features['dead'] = 0

        # features['deadEnd'] = 0
        if len(successor.getLegalActions(self.index)) <= 1:
            features['deadEnd'] = 1
        else:
            features['deadEnd'] = 0

        return features

    def getWeights(self, gameState, action):
        if self.checkScare:
            return {'successorScore': 150, 'distanceToFood': -10}
        return {'successorScore': 150, 'distanceToFood': -5, 'reverse': -3, 'distanceToEntry': -5,'dead': -200, 'deadEnd': -1, 'eatenCap': 200}

class prophet(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.restTime = 4
        self.timeToWalk = 1
        self.waitTime = 1
        self.firstTime = True
        self.isRed = True
        self.foodP = None
        self.Timer = 0
        self.turnOn = True
        self.last = 10
        self.weights = util.Counter()
        self.weights['numInvaders'] = -1000
        self.weights['ondefence'] = 100
        self.weights['distanceToPacman'] = -300
        self.weights['toNextFood'] = -50
        self.weights['toNearestFood'] = -10
        self.weights['toFarestFood'] = -10
        self.weights['reverse'] = -2
        self.weights['stop'] = -100
        self.weights['oneAwaySuperPacman'] = -1000
        self.weights['numLegalActions'] = 5
        self.weights['goDeep'] = -3

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def getQValue(self, state, action):
        total = 0
        weights = self.getWeights()
        features = self.getFeatures(state, action)
        for feature in features:
            total += features[feature] * weights[feature]
        return total

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        previousState = self.getPreviousObservation()
        foods1 = self.getFoodYouAreDefending(previousState)
        foods2 = self.getFoodYouAreDefending(gameState)
        foodsC = []
        foodsP = []
        for x in range(foods2.width):
            for y in range(foods2.height):
                if foods2[x][y]:
                    foodsC.append((x, y))
        for x in range(foods1.width):
            for y in range(foods1.height):
                if foods1[x][y]:
                    foodsP.append((x, y))
        
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemyPacmenAll = [a for a in enemies if a.isPacman]  # no matter seen or not
        enemyPacmen = [a for a in enemies if a.isPacman and a.getPosition() != None]  # seen ones
        features['numInvaders'] = len(enemyPacmenAll)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0
        if len(enemyPacmen) > 0:
            features['distanceToPacman'] = min(
                [self.getMazeDistance(myPos, pacman.getPosition()) for pacman in enemyPacmen])
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

		# find the last eaten food and go near the next defending food which is the nearest one to the eaten food.
        if len(foodsC) != len(foodsP):
            for foodP in foodsP:
                if foodP not in foodsC:
                    self.foodP = foodP
                else:
                    continue
        if self.foodP != None:
            (_, nextfood) = min([(self.getMazeDistance(self.foodP, foodC), foodC) for foodC in foodsC])
            features['toNextFood'] = self.getMazeDistance(myPos, nextfood)

		# always near some defending food
        features['toNearestFood'] = min([self.getMazeDistance(myPos, food) for food in foodsC])
        if self.last > 0 and not self.turnOn:
            features['toFarestFood'] = self.getMazeDistance(myPos, self.farFood)
            features['toNearestFood'] = 0

		# when we are scared, we still keep one distance from enenypacman
        features['oneAwaySuperPacman'] = 0
        if not myState.isPacman and myState.scaredTimer > 0:
            for i in self.getOpponents(successor):
                if successor.getAgentState(i).isPacman and successor.getAgentPosition(i) != None:
                    if self.getMazeDistance(myPos, successor.getAgentPosition(i)) == 1:
                        features['oneAwaySuperPacman'] = 1
                    if self.getMazeDistance(myPos, successor.getAgentPosition(i)) <= 2:
                        features['toNearestFood'] = 0
                        features['toNextFood'] = 0
                        features['toFarestFood'] = 0

						# if enemypacman tries to eat ourghost, we lead them to go deep in our territory
						# and avoid dead end in the meantime.
                        features['goDeep'] = self.getMazeDistance(myPos, self.start)
                        features['numLegalActions'] = len(successor.getLegalActions(self.index))
        return features
    
    def getWeights(self):
        return self.weights
    
    def maxQvalue(self, state):
        value = 0
        actions = state.getLegalActions(self.index)
        if len(actions) != 0:
            (value, action) = max([(self.getQValue(state, action), action) for action in actions])
        return (value, action)
    
    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
			# Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    
    def chooseAction(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        enemyPacmen = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if (self.timeToWalk > 0 or self.waitTime > 0) and len(enemyPacmen) == 0:
            map_height = state.data.layout.height
            map_width = state.data.layout.width
            if state.isOnRedTeam(self.index):
				# we are red, enemy is blue team
				# enemy position
                opponent1birth = (map_width - 2, map_height - 3)
                opponent3birth = (map_width - 2, map_height - 2)

				# our position, plz ignore the bad name
                opponent0birth = (1, 2)
                opponent2birth = (1, 1)
            else:
				# we are blue, enemy is red team
                self.isRed = False
				# enemy position
                opponent0birth = (1, 2)
                opponent2birth = (1, 1)

				# our position, plz ignore the bad name
                opponent1birth = (map_width - 2, map_height - 3)
                opponent3birth = (map_width - 2, map_height - 2)
            if self.isRed:
                if self.firstTime:
                    defendingFoodlist = self.getFoodYouAreDefending(state).asList()
                    (distanceToOurFood, self.nearestFoodForEnemy) = min(
                        [(self.getMazeDistance(opponent1birth, pos), pos) for pos in defendingFoodlist])
                    ourDistanceToOurFood = self.getMazeDistance(opponent2birth, self.nearestFoodForEnemy)
                    difference = distanceToOurFood - ourDistanceToOurFood
                    self.timeToWalk = ourDistanceToOurFood - 4
                    if difference >= 0:
                        self.waitTime = distanceToOurFood - self.timeToWalk
                    else:
                        self.waitTime = -1
                    self.firstTime = False
                
                if self.timeToWalk > 0:
                    actions = state.getLegalActions(self.index)
                    bestDist = 9999
                    for action in actions:
                        successor = self.getSuccessor(state, action)
                        pos2 = successor.getAgentPosition(self.index)
                        dist = self.getMazeDistance(self.nearestFoodForEnemy, pos2)
                        if dist < bestDist:
                            bestAction = action
                            bestDist = dist
                    self.timeToWalk -= 1
                    return bestAction
                elif self.waitTime > 0:
                    self.waitTime -= 1
                    return Directions.STOP
            else:
                if self.firstTime:
                    defendingFoodlist = self.getFoodYouAreDefending(state).asList()
                    (distanceToOurFood, self.nearestFoodForEnemy) = min(
                        [(self.getMazeDistance(opponent0birth, pos), pos) for pos in defendingFoodlist])
                    ourDistanceToOurFood = self.getMazeDistance(opponent3birth, self.nearestFoodForEnemy)
                    difference = distanceToOurFood - ourDistanceToOurFood
                    self.timeToWalk = ourDistanceToOurFood - 4
                    if difference >= 0:
                        self.waitTime = distanceToOurFood - self.timeToWalk
                    else:
                        self.waitTime = -1
                    self.firstTime = False
                
                if self.timeToWalk > 0:
                    actions = state.getLegalActions(self.index)
                    bestDist = 9999
                    for action in actions:
                        successor = self.getSuccessor(state, action)
                        pos2 = successor.getAgentPosition(self.index)
                        dist = self.getMazeDistance(self.nearestFoodForEnemy, pos2)
                        if dist < bestDist:
                            bestAction = action
                            bestDist = dist
                    self.timeToWalk -= 1
                    return bestAction
                elif self.waitTime > 0:
                    self.waitTime -= 1
                    return Directions.STOP
        elif self.restTime > 0 and len(enemyPacmen) == 0:
            actions = state.getLegalActions(self.index)
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(state, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.nearestFoodForEnemy, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            self.restTime -= 1
            return bestAction

		# we have a timer, every 5-action, if we can't find the enemypacman, we leave the nearest food
		# and go to the farthest food which it is when the time is up, we go to that direction for 9 steps
		# no matter we reach it or not, then we start our timer again, if we again can't find enemypacman
		# within 5-action, we move to the farthest food which it is for now.
		# This whole mechanism is like touring aroud our territory if we can't find enemy within certain time
        if self.turnOn:
            self.Timer += 1
        if self.Timer % 5 == 0:
            self.turnOn = False
        if not self.turnOn and self.last == 10:
            myp = state.getAgentPosition(self.index)
            defendingFoodlist = self.getFoodYouAreDefending(state).asList()
            (maxDist, self.farFood) = max([(self.getMazeDistance(myp, p), p) for p in defendingFoodlist])
        if not self.turnOn and self.last == 9:
            myp = state.getAgentPosition(self.index)
            defendingFoodlist = self.getFoodYouAreDefending(state).asList()
            (maxDist, self.farFood) = max([(self.getMazeDistance(myp, p), p) for p in defendingFoodlist])
        action = self.maxQvalue(state)[1]
        if not self.turnOn:
            self.last -= 1
            if self.last == 0:
                self.last = 9
                self.turnOn = True
        return action