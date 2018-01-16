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

## Functions which execute their arguments
from pandas._libs.hashtable import na_sentinel

from matplotlib import pyplot as plt


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)


def if_then_else(condition, out1, out2):
    if condition():
        return partial(progn, out1)
    else:
        return partial(progn, out2)


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

    def snakeHasCollided(self):
        self.hit = False
        if self.body[0][0] == 0 or self.body[0][0] == (YSIZE - 1) or self.body[0][1] == 0 or self.body[0][1] == (
                XSIZE - 1): self.hit = True
        if self.body[0] in self.body[1:]: self.hit = True
        return (self.hit)

    def sense_wall_ahead(self):
        self.getAheadLocation()
        return (self.ahead[0] == 0 or self.ahead[0] == (YSIZE - 1) or self.ahead[1] == 0 or self.ahead[1] == (
                XSIZE - 1))

    def sense_food_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.food

    def sense_tail_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.body

    def if_food_ahead(self, out1, out2):
        return if_then_else(self.sense_food_ahead, out1, out2)

    def if_wall_ahead(self, out1, out2):
        return if_then_else(self.sense_wall_ahead, out1, out2)

    def if_tail_ahead(self, out1, out2):
        return if_then_else(self.sense_tail_ahead, out1, out2)


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
    raw_input("Press to continue...")

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

    return totalScore,


def main():
    global snake
    global pset

    NUMBER_OF_RUNS = 2
    POPULATION_SIZE = 200
    MATE_RATE = 0.5
    MUTATION_RATE = 0.1
    GENERATIONS = 10

    logs = []

    for i in range(NUMBER_OF_RUNS):
        pop = toolbox.population(n=POPULATION_SIZE)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(lambda ind: ind.height)
        stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)

        sm, log = algorithms.eaSimple(pop, toolbox, MATE_RATE, MUTATION_RATE, GENERATIONS, stats, halloffame=hof,
                                      verbose=True)

        print 'RUN', i

        logs.append(log)

    #
    # # displayStrategyRun(hof[0])
    #
    logs_statistics(logs)

    return pop, hof, stats, log


## THIS IS WHERE YOUR CORE EVOLUTIONARY ALGORITHM WILL GO #
INIT_MIN_DEPTH = 1
INIT_MAX_DEPTH = 5
MUTATE_MIN_DEPTH = 0
MUTATE_MAX_DEPTH = 2
TOURNAMENT_SIZE = 10

pset = gp.PrimitiveSet("MAIN", 0)
pset.addPrimitive(snake.if_food_ahead, 2)
pset.addPrimitive(snake.if_wall_ahead, 2)
pset.addPrimitive(snake.if_tail_ahead, 2)
pset.addPrimitive(prog2, 2)
pset.addPrimitive(prog3, 3)

pset.addTerminal(snake.changeDirectionDown)
pset.addTerminal(snake.changeDirectionLeft)
pset.addTerminal(snake.changeDirectionRight)
pset.addTerminal(snake.changeDirectionUp)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

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
    score = runGame(routine)
    return score


toolbox.register("evaluate", evalSnake)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=MUTATE_MIN_DEPTH, max_=MUTATE_MAX_DEPTH)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

from scoop import futures

toolbox.register("map", futures.map)


## THE FOLLOWING FUNCTIONS EVALUATE THE PERFORMANCE OF THE ALGORITHM

def analyse(logs, hof):
    statistics = logs_statistics(logs)


def plotstuff(log):
    gen = log.select("gen")
    fit_mins = log.chapters["fitness"].select("avg")
    size_avgs = log.chapters["size"].select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Average Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()


def logs_statistics(logs):
    from pandas import DataFrame

    # Convert logbook to numpy arrays
    log_gen = numpy.array([log.select('gen') for log in logs])
    log_nevals = numpy.array([log.select('nevals') for log in logs])
    log_fitness_avg = numpy.array([log.chapters['fitness'].select('avg') for log in logs])
    log_fitness_max = numpy.array([log.chapters['fitness'].select('max') for log in logs])
    log_fitness_min = numpy.array([log.chapters['fitness'].select('min') for log in logs])
    log_fitness_std = numpy.array([log.chapters['fitness'].select('std') for log in logs])
    log_size_avg = numpy.array([log.chapters['size'].select('avg') for log in logs])
    log_size_max = numpy.array([log.chapters['size'].select('max') for log in logs])
    log_size_min = numpy.array([log.chapters['size'].select('min') for log in logs])
    log_size_std = numpy.array([log.chapters['size'].select('std') for log in logs])

    # summarise statistics for each generation
    data_dic = {}
    data_dic['gen'] = numpy.mean(log_gen, 0)
    data_dic['nevals'] = numpy.mean(log_nevals, 0)

    data_dic['fitness_avg'] = numpy.mean(log_fitness_avg, 0)
    data_dic['fitness_max'] = numpy.max(log_fitness_max, 0)
    data_dic['fitness_min'] = numpy.min(log_fitness_min, 0)
    data_dic['fitness_std'] = numpy.sqrt(numpy.sum(log_fitness_std ** 2, 0))  # pooled std

    data_dic['size_avg'] = numpy.mean(log_size_avg, 0)
    data_dic['size_max'] = numpy.max(log_size_max, 0)
    data_dic['size_min'] = numpy.min(log_size_min, 0)
    data_dic['size_std'] = numpy.sqrt(numpy.sum(log_size_std ** 2, 0))  # pooled std

    df = DataFrame(data_dic)

    cols = ['gen', 'nevals', 'fitness_avg', 'fitness_max', 'fitness_min', 'fitness_std', 'size_avg', 'size_max',
            'size_min', 'size_std']
    df.to_csv('eggs.csv', sep=',', columns=cols)

    df[['gen', 'fitness_avg', 'fitness_std']].plot(x='gen', yerr='fitness_std')
    plt.show(kind='box')

    print df
    return
    keys = ['gen', 'nevals'] + [j + '_' + i for j in sorted(logs[0].chapters.keys())
                                for i in sorted(logs[0].chapters['fitness'][0].keys())]

    values = [{k: 0 for k in keys} for j in range(len(logs[0]))]

    for log in logs:
        for i in range(len(log)):

            values[i]['gen'] += log[i]['gen']
            values[i]['nevals'] += log[i]['nevals']

            for key in log.chapters.keys():
                for subkey in log.chapters[key][i].keys():
                    if subkey is 'std':
                        values[i][key + '_' + subkey] += log.chapters[key][i][subkey] ** 2
                    elif subkey is 'max':
                        values[i][key + '_' + subkey] = max(log.chapters[key][i][subkey], values[i][key + '_' + subkey])
                    elif subkey is 'min':
                        values[i][key + '_' + subkey] = min(log.chapters[key][i][subkey], values[i][key + '_' + subkey])
                    else:
                        values[i][key + '_' + subkey] += log.chapters[key][i][subkey]

    for dic in values:
        for key in dic.keys():
            if key is 'std':
                dic[key] = numpy.sqrt(len(logs) / len(logs))
            elif key is 'max':
                pass
            elif key is 'min':
                pass
            else:
                dic[key] /= len(logs)

    with open('eggs.csv', 'wb') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=keys)
        writer.writeheader()
        writer.writerows(values)

    return values


if __name__ == '__main__':
    main()
