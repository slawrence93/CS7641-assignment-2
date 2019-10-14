# CS7641-assignment-2

make sure jython is installed

`four_peaks.py` will run all of the random optimization algorithms for the four peaks problem
`count_ones.py` will run all of the random optimization algorithms for the count ones/max ones problem
`knapsack.py` will run all of the random optimization algorithms for the knapsack problem

all of the programs will output the data used in the charts to a csv file with the name `<optimization problem name>_[rhc|sa|ga|mm].csv` in the output folder (however you may have to delete the existing csv files for it work, not sure)

to see how the random optimization algorithms fare when training weights for a neural network run the below programs and data will be printed to the console:
`nn_rhc.py`
`nn_sa.py`
`nn_ga.py`
