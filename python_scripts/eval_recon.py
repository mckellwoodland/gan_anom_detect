"""
Evaluate reconstructions patch-wise.

Corresponding original and reconstructed images must have the same name.
Code is built for 2-dimensional grayscale PNG images.
"""

# Imports
import argparse
import imageio.v2 as imageio
import os
import pandas as pd
import tqdm
from os import path
from scipy import stats
from skimage import metrics

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-o','--orig_dir',type=str,help='Path to the directory containing the original images.')
required.add_argument('-op','--out_path',type=str,help='Path to csv file to write distances to.')
required.add_argument('-r','--recon_dir',type=str,help='Path to the directory containing the reconstructed images.')

optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-d','--distance',type=str,default='MSE',help='One of [MSE, WD] for mean-squared error and Wasserstein distance.\
                            Defaults to MSE.')
args = parser.parse_args()

# Main Code
if __name__=="__main__":
    dists = {'filename':[], 'dist':[]}

    # Iterate through all the original images.
    for fname in tqdm.tqdm(os.listdir(args.orig_dir)):
        # Read in the images.
        orig_path = path.join(args.orig_dir, fname)
        recon_path = path.join(args.recon_dir, fname)
        orig = imageio.imread(orig_path, 'PNG-PIL', pilmode='L')
        recon = imageio.imread(recon_path, 'PNG-PIL', pilmode='L')

        # Evaluate reconstructions
        if args.distance == "MSE":
            dist = metrics.mean_squared_error(orig, recon)
        elif args.distance == "WD":
            orig = orig.flatten()
            recon = recon.flatten()
            dist = stats.wasserstein_distance(orig, recon)
        dists['filename'].append(fname)
        dists['dist'].append(dist)

    # Write out distances to CSV.
    dist_df = pd.DataFrame(dists)
    dist_df.to_csv(args.out_path, index=False)
