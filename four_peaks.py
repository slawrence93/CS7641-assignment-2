from array import array

import os
import sys

sys.path.append("./ABAGAIL.jar")

from dist import DiscreteUniformDistribution, DiscreteDependencyTree
from opt import RandomizedHillClimbing, SimulatedAnnealing, GenericHillClimbingProblem, DiscreteChangeOneNeighbor
from opt.example import FourPeaksEvaluationFunction
from opt.ga import StandardGeneticAlgorithm, DiscreteChangeOneMutation, SingleCrossOver, GenericGeneticAlgorithmProblem
from opt.prob import MIMIC, GenericProbabilisticOptimizationProblem
from shared import FixedIterationTrainer
import time
import itertools


def run_four_peaks_experiments():
    OUTPUT_DIRECTORY = './output'

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    N = 200
    T = N/5
    fill = [2] * N
    ranges = array('i', fill)
    ef = FourPeaksEvaluationFunction(T)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = SingleCrossOver()
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    max_iter = 5000
    outfile = OUTPUT_DIRECTORY + '/four_peaks_{}_log.csv'

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
        f.write('iteration,cooling_value,fitness,elapsed_time\n')
    for cooling_value in (.19, .38, .76, .95):
        for it in range(0, max_iter, 10):
            sa = SimulatedAnnealing(1E11, cooling_value, hcp)
            fit = FixedIterationTrainer(sa, it)
            start_time = time.clock()
            fit.train()
            elapsed_time = time.clock() - start_time
            # fevals = ef.fevalss
            score = ef.value(sa.getOptimal())
            data = '{},{},{},{}\n'.format(it, cooling_value, score, elapsed_time)
            print(data)
            with open(filename, 'a') as f:
                f.write(data)

    # Genetic Algorithm
    filename = outfile.format('ga')
    with open(filename, 'w') as f:
        f.write('iteration,population_size,to_mate,to_mutate,fitness,time\n')
    for population_size, to_mate, to_mutate in itertools.product([200], [25, 50, 75, 100], [10, 20, 30, 40]):
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
        f.write('iterations,samples,to_keep,fitness,time\n')
    for samples, to_keep, m in itertools.product([200], [20], [0.1, 0.3, 0.5, 0.7, 0.9]):
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
    run_four_peaks_experiments()
