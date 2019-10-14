from array import array
from random import Random
import sys

sys.path.append("./ABAGAIL.jar")
import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random
from dist import DiscreteUniformDistribution, DiscreteDependencyTree
from opt import RandomizedHillClimbing, SimulatedAnnealing, GenericHillClimbingProblem, DiscreteChangeOneNeighbor
from opt.example import KnapsackEvaluationFunction
from opt.ga import StandardGeneticAlgorithm, DiscreteChangeOneMutation, SingleCrossOver, GenericGeneticAlgorithmProblem, \
    UniformCrossOver
from opt.prob import MIMIC, GenericProbabilisticOptimizationProblem
from shared import FixedIterationTrainer
import time
import itertools


def run_knapsack_experiments():
    OUTPUT_DIRECTORY = './output'
    # Random number generator */
    random = Random()
    # The number of items
    NUM_ITEMS = 40
    # The number of copies each
    COPIES_EACH = 4
    # The maximum weight for a single element
    MAX_WEIGHT = 50
    # The maximum volume for a single element
    MAX_VOLUME = 50
    # The volume of the knapsack
    KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

    # create copies
    fill = [COPIES_EACH] * NUM_ITEMS
    copies = array('i', fill)

    # create weights and volumes
    fill = [0] * NUM_ITEMS
    weights = array('d', fill)
    volumes = array('d', fill)
    for i in range(0, NUM_ITEMS):
        weights[i] = random.nextDouble() * MAX_WEIGHT
        volumes[i] = random.nextDouble() * MAX_VOLUME


    # create range
    fill = [COPIES_EACH + 1] * NUM_ITEMS
    ranges = array('i', fill)

    ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = UniformCrossOver()
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    max_iter = 5000
    outfile = OUTPUT_DIRECTORY + '/knapsack_{}_log.csv'

    # Randomized Hill Climber
    filename = outfile.format('rhc')
    with open(filename, 'w') as f:
        f.write('iterations,fitness,time\n')
    for it in range(0, max_iter, 10):
        rhc = RandomizedHillClimbing(hcp)
        fit = FixedIterationTrainer(rhc, it)
        start_time = time.clock()
        fit.train()
        elapsed_time = time.clock() - start_time
        # fevals = ef.fevals
        score = ef.value(rhc.getOptimal())
        data = '{},{},{}\n'.format(it, score, elapsed_time)
        print(data)
        with open(filename, 'a') as f:
            f.write(data)

    # Simulated Annealing
    filename = outfile.format('sa')
    with open(filename, 'w') as f:
        f.write('iteration,cooling_value,fitness,time\n')
    for cooling_value in (.19, .38, .76, .95):
        for it in range(0, max_iter, 10):
            sa = SimulatedAnnealing(200, cooling_value, hcp)
            fit = FixedIterationTrainer(sa, it)
            start_time = time.clock()
            fit.train()
            elapsed_time = time.clock() - start_time
            # fevals = ef.fevals
            score = ef.value(sa.getOptimal())
            data = '{},{},{},{}\n'.format(it, cooling_value, score, elapsed_time)
            print(data)
            with open(filename, 'a') as f:
                f.write(data)

    # Genetic Algorithm
    filename = outfile.format('ga')
    with open(filename, 'w') as f:
        f.write('iteration,population_size,to_mate,to_mutate,fitness,time\n')
    for population_size, to_mate, to_mutate in itertools.product([200], [110, 120, 130, 140, 150], [2, 4, 6, 8]):
        for it in range(0, max_iter, 10):
            ga = StandardGeneticAlgorithm(population_size, to_mate, to_mutate, gap)
            fit = FixedIterationTrainer(ga, it)
            start_time = time.clock()
            fit.train()
            elapsed_time = time.clock() - start_time
            # fevals = ef.fevals
            score = ef.value(ga.getOptimal())
            data = '{},{},{},{},{},{}\n'.format(it, population_size, to_mate, to_mutate, score, elapsed_time)
            print(data)
            with open(filename, 'a') as f:
                f.write(data)

    # MIMIC
    filename = outfile.format('mm')
    with open(filename, 'w') as f:
        f.write('iterations,samples,to_keep,m,fitness,time\n')
    for samples, to_keep, m in itertools.product([200], [100], [0.1, 0.3, 0.5, 0.7, 0.9]):
        for it in range(0, 500, 10):
            df = DiscreteDependencyTree(m, ranges)
            pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
            mm = MIMIC(samples, 20, pop)
            fit = FixedIterationTrainer(mm, it)
            start_time = time.clock()
            fit.train()
            elapsed_time = time.clock() - start_time
            # fevals = ef.fevals
            score = ef.value(mm.getOptimal())
            data = '{},{},{},{},{},{}\n'.format(it, samples, to_keep, m, score, elapsed_time)
            print(data)
            with open(filename, 'a') as f:
                f.write(data)


if __name__ == '__main__':
    run_knapsack_experiments()
