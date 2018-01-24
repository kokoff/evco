# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
import curses
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import operator
from functools import partial
import numpy


# Helper functions
def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


def if_then_else(condition, out1, out2):
    if condition():
        out1()
    else:
        out2()


S_RIGHT, S_LEFT, S_UP, S_DOWN = 0, 1, 2, 3
XSIZE, YSIZE = 14, 14
NFOOD = 1  # NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)


# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
    global S_RIGHT, S_LEFT, S_UP, S_DOWN
    global XSIZE, YSIZE

    def __init__(self):
        self.direction = S_RIGHT
        self.body = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.score = 0
        self.ahead = []
        self.food = []

    def _reset(self):
        self.direction = S_RIGHT
        self.body[:] = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.score = 0
        self.ahead = []
        self.food = []

    def getAheadLocation(self):
        self.ahead = [self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1),
                      self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)]

    def getLeftLocation(self):
        return [self.body[0][0] + (self.direction == S_LEFT and 1) + (self.direction == S_RIGHT and -1),
                self.body[0][1] + (self.direction == S_UP and -1) + (self.direction == S_DOWN and 1)]

    def getRightLocation(self):
        return [self.body[0][0] + (self.direction == S_RIGHT and 1) + (self.direction == S_LEFT and -1),
                self.body[0][1] + (self.direction == S_DOWN and -1) + (self.direction == S_UP and 1)]

    def get_ahead_direction(self):
        return self.direction

    def get_left_direction(self):
        dic = {S_UP: S_LEFT, S_DOWN: S_RIGHT, S_LEFT: S_DOWN, S_RIGHT: S_UP}
        return dic[self.direction]

    def get_right_direction(self):
        dic = {S_UP: S_RIGHT, S_DOWN: S_LEFT, S_LEFT: S_UP, S_RIGHT: S_DOWN}
        return dic[self.direction]

    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead)

    ## You are free to define more sensing options to the snake

    def changeDirectionUp(self):
        self.direction = S_UP

    def changeDirectionRight(self):
        self.direction = S_RIGHT

    def changeDirectionDown(self):
        self.direction = S_DOWN

    def changeDirectionLeft(self):
        self.direction = S_LEFT

    def doNothing(self):
        pass

    def snakeHasCollided(self):
        self.hit = False
        if self.body[0][0] == 0 or self.body[0][0] == (YSIZE - 1) or self.body[0][1] == 0 or self.body[0][1] == (
                XSIZE - 1): self.hit = True
        if self.body[0] in self.body[1:]: self.hit = True
        return (self.hit)

    # basic sensing
    def sense_wall_in_square(self, square):
        return square[0] == 0 or square[0] == (YSIZE - 1) or square[1] == 0 or square[1] == (XSIZE - 1)

    def sense_tail_in_square(self, square):
        return square in self.body[2:]

    def get_adjecent_square(self, square, direction):
        return [square[0] + (direction == S_DOWN and 1) + (direction == S_UP and -1),
                square[1] + (direction == S_LEFT and -1) + (direction == S_RIGHT and 1)]

    def sense_wall_in_adjecent_square(self, direction):
        square = self.get_adjecent_square(self.body[0], direction)
        return self.sense_wall_in_square(square)

    def sense_tail_in_adjecent_square(self, direction):
        square = self.get_adjecent_square(self.body[0], direction)
        return self.sense_tail_in_square(square)

    def sense_danger_in_adjecent_square(self, direction):
        square = self.get_adjecent_square(self.body[0], direction)
        return self.sense_wall_in_square(square) or self.sense_tail_in_square(square)

    def sense_food_in_line(self, direction):
        if direction == S_DOWN:
            line = [[i, self.body[0][1]] for i in range(self.body[0][0] + 1, YSIZE)]
        if direction == S_UP:
            line = [[i, self.body[0][1]] for i in range(0, self.body[0][0])]
        if direction == S_RIGHT:
            line = [[self.body[0][0], i] for i in range(self.body[0][1] + 1, XSIZE)]
        if direction == S_LEFT:
            line = [[self.body[0][0], i] for i in range(0, self.body[0][1])]

        return any([i for i in line if i in self.food])

    # RELATIVE TO SNAKE SENSING-----------------------------------------------------------------------------

    # food
    def sense_food_ahead(self):
        return self.sense_food_in_line(self.direction)

    def sense_food_on_left(self):
        return self.sense_food_in_line(self.get_left_direction())

    def sense_food_on_right(self):
        return self.sense_food_in_line(self.get_right_direction())

    def if_food_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_food_ahead, out1, out2)

    def if_food_on_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_on_left, out1, out2)

    def if_food_on_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_on_right, out1, out2)

    # wall
    def sense_wall_ahead(self):
        direction = self.get_ahead_direction()
        return self.sense_wall_in_adjecent_square(direction)

    def sense_wall_on_left(self):
        direction = self.get_left_direction()
        return self.sense_wall_in_adjecent_square(direction)

    def sense_wall_on_right(self):
        direction = self.get_right_direction()
        return self.sense_wall_in_adjecent_square(direction)

    def if_wall_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_wall_ahead, out1, out2)

    def if_wall_on_left(self, out1, out2):
        return partial(if_then_else, self.sense_wall_on_left, out1, out2)

    def if_wall_on_right(self, out1, out2):
        return partial(if_then_else, self.sense_wall_on_right, out1, out2)

    # tail
    def sense_tail_ahead(self):
        direction = self.get_ahead_direction()
        return self.sense_tail_in_adjecent_square(direction)

    def sense_tail_on_left(self):
        direction = self.get_left_direction()
        return self.sense_tail_in_adjecent_square(direction)

    def sense_tail_on_right(self):
        direction = self.get_right_direction()
        return self.sense_tail_in_adjecent_square(direction)

    def if_tail_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_tail_ahead, out1, out2)

    def if_tail_on_left(self, out1, out2):
        return partial(if_then_else, self.sense_tail_on_left, out1, out2)

    def if_tail_on_right(self, out1, out2):
        return partial(if_then_else, self.sense_tail_on_right, out1, out2)

    # ------------------------------------------------------------------------------------------------------

    # RELATIVE TO GRID SENSING------------------------------------------------------------------------------

    # food
    def if_food_up(self, out1, out2):
        cond = partial(self.sense_food_in_line, S_UP)
        return partial(if_then_else, cond, out1, out2)

    def if_food_down(self, out1, out2):
        cond = partial(self.sense_food_in_line, S_DOWN)
        return partial(if_then_else, cond, out1, out2)

    def if_food_left(self, out1, out2):
        cond = partial(self.sense_food_in_line, S_LEFT)
        return partial(if_then_else, cond, out1, out2)

    def if_food_right(self, out1, out2):
        cond = partial(self.sense_food_in_line, S_RIGHT)
        return partial(if_then_else, cond, out1, out2)

    # wall
    def if_wall_up(self, out1, out2):
        cond = partial(self.sense_wall_in_adjecent_square, S_UP)
        return partial(if_then_else, cond, out1, out2)

    def if_wall_down(self, out1, out2):
        cond = partial(self.sense_wall_in_adjecent_square, S_DOWN)
        return partial(if_then_else, cond, out1, out2)

    def if_wall_left(self, out1, out2):
        cond = partial(self.sense_wall_in_adjecent_square, S_LEFT)
        return partial(if_then_else, cond, out1, out2)

    def if_wall_right(self, out1, out2):
        cond = partial(self.sense_wall_in_adjecent_square, S_RIGHT)
        return partial(if_then_else, cond, out1, out2)

    # tail
    def if_tail_up(self, out1, out2):
        cond = partial(self.sense_tail_in_adjecent_square, S_UP)
        return partial(if_then_else, cond, out1, out2)

    def if_tail_down(self, out1, out2):
        cond = partial(self.sense_tail_in_adjecent_square, S_DOWN)
        return partial(if_then_else, cond, out1, out2)

    def if_tail_left(self, out1, out2):
        cond = partial(self.sense_tail_in_adjecent_square, S_LEFT)
        return partial(if_then_else, cond, out1, out2)

    def if_tail_right(self, out1, out2):
        cond = partial(self.sense_tail_in_adjecent_square, S_RIGHT)
        return partial(if_then_else, cond, out1, out2)

    # ------------------------------------------------------------------------------------------------------

    # DIRECTION OF TRAVEL SENSING------------------------------------------------------------------------------
    def is_direction(self, direction):
        return self.direction == direction

    def if_moving_up(self, out1, out2):
        cond = partial(self.is_direction, S_UP)
        return partial(if_then_else, cond, out1, out2)

    def if_moving_down(self, out1, out2):
        cond = partial(self.is_direction, S_DOWN)
        return partial(if_then_else, cond, out1, out2)

    def if_moving_left(self, out1, out2):
        cond = partial(self.is_direction, S_LEFT)
        return partial(if_then_else, cond, out1, out2)

    def if_moving_right(self, out1, out2):
        cond = partial(self.is_direction, S_RIGHT)
        return partial(if_then_else, cond, out1, out2)
    # ------------------------------------------------------------------------------------------------------


# This function places a food item in the environment
def placeFood(snake):
    food = []
    while len(food) < NFOOD:
        potentialfood = [random.randint(1, (YSIZE - 2)), random.randint(1, (XSIZE - 2))]
        if not (potentialfood in snake.body) and not (potentialfood in food):
            food.append(potentialfood)
    snake.food = food  # let the snake know where the food is
    return (food)


snake = SnakePlayer()


# This outline function is the same as runGame (see below). However,
# it displays the game graphically and thus runs slower
# This function is designed for you to be able to view and assess
# your strategies, rather than use during the course of evolution
def displayStrategyRun(individual):
    global snake
    global pset

    routine = gp.compile(individual, pset)

    curses.initscr()
    win = curses.newwin(YSIZE, XSIZE, 0, 0)
    win.keypad(1)
    curses.noecho()
    curses.curs_set(0)
    win.border(0)
    win.nodelay(1)
    win.timeout(120)

    snake._reset()
    food = placeFood(snake)

    for f in food:
        win.addch(f[0], f[1], '@')

    timer = 0
    collided = False
    while not collided and not timer == ((2 * XSIZE) * YSIZE):

        # Set up the display
        win.border(0)
        win.addstr(0, 2, 'Score : ' + str(snake.score) + ' ')
        win.getch()

        ## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
        routine()

        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            for f in food: win.addch(f[0], f[1], ' ')
            food = placeFood(snake)
            for f in food: win.addch(f[0], f[1], '@')
            timer = 0
        else:
            last = snake.body.pop()
            win.addch(last[0], last[1], ' ')
            timer += 1  # timesteps since last eaten
        win.addch(snake.body[0][0], snake.body[0][1], 'o')

        collided = snake.snakeHasCollided()
        hitBounds = (timer == ((2 * XSIZE) * YSIZE))

    curses.endwin()

    print collided
    print hitBounds
    # raw_input("Press to continue...")

    return snake.score,


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.
def runGame(routine):
    global snake

    totalScore = 0

    snake._reset()
    food = placeFood(snake)
    timer = 0
    elapsed = 0
    while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:

        ## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
        routine()

        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            food = placeFood(snake)
            timer = 0
        else:
            snake.body.pop()
            timer += 1  # timesteps since last eaten

        totalScore += snake.score
        elapsed += 1
    timedOut = timer == XSIZE * YSIZE
    return snake.score, elapsed, totalScore, timedOut


def main():
    global snake
    global pset

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_score = tools.Statistics(lambda ind: ind.score)
    stats_steps = tools.Statistics(lambda ind: ind.steps)
    stats_depth = tools.Statistics(lambda ind: ind.height)
    stats_size = tools.Statistics(lambda ind: len(ind))
    stats = tools.MultiStatistics(fitness=stats_fit, score=stats_score, steps=stats_steps, size=stats_size,
                                  depth=stats_depth)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    sm, log = algorithms.eaSimple(pop, toolbox, MATE_RATE, MUTATION_RATE, GENERATIONS, stats, halloffame=hof,
                                  verbose=True)

    return pop, hof, stats, log


## THIS IS WHERE YOUR CORE EVOLUTIONARY ALGORITHM WILL GO #
NUMBER_OF_RUNS = 20
NUMBER_OF_TEST_RUNS = 100

POPULATION_SIZE = 600
MATE_RATE = 0.6
MUTATION_RATE = 0.1
GENERATIONS = 400

INIT_MIN_DEPTH = 2
INIT_MAX_DEPTH = 9
LEAF_MATE_RATE = 0.1
MUTATE_MIN_DEPTH = 1
MUTATE_MAX_DEPTH = 2
TOURNAMENT_SIZE = 7
PARSIMONY_SIZE = 1.2

TREE_MAX_NODES = 150

pset = gp.PrimitiveSet("MAIN", 0)

pset.addPrimitive(snake.if_food_up, 2)
pset.addPrimitive(snake.if_food_down, 2)
pset.addPrimitive(snake.if_food_left, 2)
pset.addPrimitive(snake.if_food_right, 2)

pset.addPrimitive(snake.if_wall_up, 2)
pset.addPrimitive(snake.if_wall_down, 2)
pset.addPrimitive(snake.if_wall_left, 2)
pset.addPrimitive(snake.if_wall_right, 2)

pset.addPrimitive(snake.if_tail_up, 2)
pset.addPrimitive(snake.if_tail_down, 2)
pset.addPrimitive(snake.if_tail_left, 2)
pset.addPrimitive(snake.if_tail_right, 2)

# pset.addPrimitive(snake.if_food_ahead, 2)
# pset.addPrimitive(snake.if_food_on_left, 2)
# pset.addPrimitive(snake.if_food_on_right, 2)
#
# pset.addPrimitive(snake.if_wall_ahead, 2)
# pset.addPrimitive(snake.if_wall_on_left, 2)
# pset.addPrimitive(snake.if_wall_on_right, 2)
#
# pset.addPrimitive(snake.if_tail_ahead, 2)
# pset.addPrimitive(snake.if_tail_on_left, 2)
# pset.addPrimitive(snake.if_tail_on_right, 2)
#
# pset.addPrimitive(snake.if_moving_up, 2)
# pset.addPrimitive(snake.if_moving_down, 2)
# pset.addPrimitive(snake.if_moving_left, 2)
# pset.addPrimitive(snake.if_moving_right, 2)

pset.addPrimitive(prog2, 2)

pset.addTerminal(snake.changeDirectionDown)
pset.addTerminal(snake.changeDirectionLeft)
pset.addTerminal(snake.changeDirectionRight)
pset.addTerminal(snake.changeDirectionUp)
pset.addTerminal(snake.doNothing)


# Add attributes to PrimitiveTree so they can be used for statistics
class SnakePrimitiveTree(gp.PrimitiveTree):

    def __init__(self, content):
        self.score = 0
        self.steps = 0
        gp.PrimitiveTree.__init__(self, content)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", SnakePrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=INIT_MIN_DEPTH, max_=INIT_MAX_DEPTH)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalSnake(individual):
    # Transform the tree expression to functionnal Python code
    routine = gp.compile(individual, pset)
    # Run the generated routine
    score, elapsed, totalScore, timedOut = runGame(routine)

    # Save score and elapsed
    individual.score = score
    individual.steps = elapsed

    # Multi objective evaluation
    fitness = score + elapsed
    if timedOut:
        fitness -= XSIZE * YSIZE

    # Simple evaluation
    # fitness = score

    return fitness,


toolbox.register("evaluate", evalSnake)
toolbox.register("select", tools.selDoubleTournament, fitness_size=TOURNAMENT_SIZE, parsimony_size=PARSIMONY_SIZE,
                 fitness_first=True)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=LEAF_MATE_RATE)
toolbox.register("expr_mut", gp.genFull, min_=MUTATE_MIN_DEPTH, max_=MUTATE_MAX_DEPTH)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=len, max_value=TREE_MAX_NODES))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=TREE_MAX_NODES))

# Run in parallel
from scoop import futures

toolbox.register("map", futures.map)


## THE FOLLOWING FUNCTIONS EVALUATE THE PERFORMANCE OF THE ALGORITHM

def run_best_individual(individual):
    while True:
        print displayStrategyRun(individual)[0]
        a = raw_input("Do you want to run again [y/any_key]?")
        if a is not 'y':
            break


## Run an individual 100 times with fixed random seed output results in csv
def evaluate_best_individual(best_individual):
    try:
        from pandas import DataFrame
    except:
        return

    # evaluate best individual out of all runs
    number_of_test_runs = NUMBER_OF_TEST_RUNS
    best_stats = {'fitness': [], 'score': [], 'steps': []}

    for i in range(number_of_test_runs):
        random.seed(i)
        evalSnake(best_individual)
        random.seed()
        best_stats['fitness'].append(float(best_individual.fitness.values[0]))
        best_stats['score'].append(float(best_individual.score))
        best_stats['steps'].append(float(best_individual.steps))

    pd = DataFrame(best_stats)
    pd.to_csv('best_individual.csv', sep=',')


def evaluate_best_individuals(best_individuals):
    try:
        from pandas import DataFrame
    except:
        return

    # evaluate best individual out of all runs
    number_of_test_runs = NUMBER_OF_TEST_RUNS
    best_stats = {'fitness': [0.0 for i in range(len(best_individuals))],
                  'score': [0.0 for i in range(len(best_individuals))],
                  'steps': [0.0 for i in range(len(best_individuals))]}

    for i in range(len(best_individuals)):
        for j in range(number_of_test_runs):
            individual = best_individuals[i]

            random.seed(i)
            evalSnake(individual)
            random.seed()

            best_stats['fitness'][i] += individual.fitness.values[0]
            best_stats['score'][i] += individual.score
            best_stats['steps'][i] += individual.steps

        best_stats['fitness'][i] /= float(number_of_test_runs)
        best_stats['score'][i] /= float(number_of_test_runs)
        best_stats['steps'][i] /= float(number_of_test_runs)

    pd = DataFrame(best_stats)
    pd.to_csv('best_individuals.csv', sep=',')


def save_best_individual(best_individual):
    with open('best_individual.txt', 'w') as f:
        f.write(str(best_individual))


def save_best_individuals(best_individuals):
    individuals = [str(ind) + '\n' for ind in best_individuals]
    with open('best_individuals.txt', 'w') as f:
        f.writelines(individuals)


if __name__ == '__main__':

    best_individuals = []
    individual = None

    for i in range(NUMBER_OF_RUNS):
        print 'RUN', i

        pop, hof, stats, log = main()

        # collect best individuals
        best_individuals.append(hof[0])

        # evaluate best individuals for each run
        if individual is None or individual.fitness.values[0] < hof[0].fitness.values[0]:
            individual = hof[0]

    evaluate_best_individual(individual)
    evaluate_best_individuals(best_individuals)
    save_best_individual(individual)
    save_best_individuals(best_individuals)
    run_best_individual(individual)

    print individual
