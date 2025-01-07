"""
Perform a one-sided permutation test between two populations, where the two populations are columns in a csv file."""

# Imports.
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import tqdm

# Arguments.
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-c1','--csv1',type=str,help='Path to csv file containing the first population.')
required.add_argument('-c2','--csv2',type=str,help='Path to csv file containing the second population.')
required.add_argument('-n1','--col_name1',type=str,help='Name of column containing the first population.')
required.add_argument('-n2','--col_name2',type=str,help='Name of column containing the second population.')

args = parser.parse_args()

# Functions.
def perm_test(pop1, pop2, num_itrs=int(1e5)):
    mean_diff_true = round(np.mean(pop1) - np.mean(pop2),10)
    samples = np.array(pop1 + pop2)
    count = 0
    for i in range(num_itrs):
        np.random.shuffle(samples)
        array1, array2 = samples[:len(pop1)], samples[len(pop1):]
        mean_diff = round(np.mean(array1) - np.mean(array2),10)
        if abs(mean_diff) > mean_diff_true:
            count += 1
    return count/num_itrs

def sim_pow(pop1, pop2, n, alpha=0.05, num_sim=int(1e3), num_itrs=int(1e3)):
    mean_diff_true = np.mean(pop1) - np.mean(pop2)
    std1, std2 = np.std(pop1), np.std(pop2)
    count = 0
    for i in tqdm.tqdm(range(num_sim)):
        data1 = np.random.normal(0,std1, n)
        data2 = np.random.normal(mean_diff_true, std2, n)
        p = perm_test(pop1, pop2, num_itrs)
        if p < alpha:
            count += 1
    return count / num_sim

# Main script.
if __name__=="__main__":
    pop1 = list(pd.read_csv(args.csv1)[args.col_name1])
    pop2 = list(pd.read_csv(args.csv2)[args.col_name2])
    print("Power of calculation:", sim_pow(pop1, pop2, len(pop1)))
    print("Permutation test p-value:", perm_test(pop1,pop2))
