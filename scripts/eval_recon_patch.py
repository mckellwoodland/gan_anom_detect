"""
Evaluate reconstructions patch-wise.

Corresponding original and reconstructed images must have the same name.
Code is built for 2-dimensional grayscale PNG images.
"""

# Imports
import argparse
import imageio
import os
import tqdm
from os import path
from scipy import stats
from skimage import metrics

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
required.add_argument('-o',
                      '--orig_dir',
                      type=str,
                      help='Path to the directory containing the original images.')
required.add_argument('-r',
                      '--recon_dir',
                      type=str,
                      help='Path to the directory containing the reconstructed images.')
optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-d', 
                      '--distance', 
                      type=str, 
                      default='MSE',
                      help='One of [MSE, WD, SS] for mean-squared error, Wasserstein distance, or Structural Similarity.\
                            Defaults to MSE.')
optional.add_argument('-s',
                      '--patch_size',
                      type=int,
                      default=32,
                      help='One integer representing the dimensionality of the patch to be evaluated.\
                            Defaults to 32.')
args = parser.parse_args()

# Main Code
if __name__=="__main__":
  # Iterate through all the original images.
  for fname in tqdm.tqdm(os.listdir(args.orig_dir)):
      # Read in the images.
      orig_path = path.join(args.orig_dir, fname)
      recon_path = path.join(args.recon_dir, fname)
      orig = imageio.imread(orig_path, 'PNG-PIL', pilmode='L')
      recon = imageio.imread(recon_path, 'PNG-PIL', pilmode='L')
