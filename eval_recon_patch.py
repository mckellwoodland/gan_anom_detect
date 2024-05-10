"""
Evaluate reconstructions patch-wise.

Corresponding original and reconstructed images must have the same name.
Code is built for 2-dimensional PNG images.
"""

# Imports
import argparse
import os
from skimage import metrics, stats
# metrics.mean_squared_error, metrics.structural_similarity, stats.wasserstein_distance

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('d', 
                      '--distance', 
                      type=str, 
                      default='MSE',
                      help='One of [MSE, WD, SS] for mean-squared error, Wasserstein distance, or Structural Similarity.')
args = parser.parse_args()

