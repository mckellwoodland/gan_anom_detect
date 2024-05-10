"""
Evaluate reconstructions patch-wise.

Corresponding original and reconstructed images must have the same name.
Code is built for 2-dimensional PNG images.
"""

# Imports
import argparse
import os
from scipy import stats
from skimage import metrics

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-d', 
                      '--distance', 
                      type=str, 
                      default='MSE',
                      help='One of [MSE, WD, SS] for mean-squared error, Wasserstein distance, or Structural Similarity.\
                            Defaults to MSE.')
args = parser.parse_args()

