"""
Evaluate reconstructions patch-wise.

Corresponding original and reconstructed images must have the same name.
Code is built for 2-dimensional grayscale PNG images.
"""

# Imports
import argparse
import imageio.v2 as imageio
import numpy as np
import os
import pandas as pd
import random
import tqdm
from os import path
from scipy import stats
from skimage import metrics, measure

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
optional.add_argument('-ps','--patch_size',type=int,default=32,help='Size of the patches when the region argument is "patch".\
                                                                    Default: 32.')
optional.add_argument('-pp','--patch_path',type=str,default=None,help='Folder to save the highest-scoring patches and their reconstructions to.\
                                                                       Default to None. \
                                                                       If None, no patches will be saved.')
optional.add_argument('-reg','--region',type=str,default='full',help='The region of the image that the reconstruction metric should be calculated on. \
                                                                    Options: ["full","body","patch"].\
                                                                    Default: "full".')
args = parser.parse_args()

# Main Code
if __name__=="__main__":
    dists = {'filename':[], 'dist':[]}

    # Iterate through all the original images.
    filenames = os.listdir(args.orig_dir)
    random.shuffle(filenames)
    for fname in tqdm.tqdm(filenames):
        # Read in the images.
        orig_path = path.join(args.orig_dir, fname)
        recon_path = path.join(args.recon_dir, fname)
        orig = imageio.imread(orig_path, 'PNG-PIL', pilmode='L')
        recon = imageio.imread(recon_path, 'PNG-PIL', pilmode='L')
        if args.region=="body":
            width, height = orig.shape
            thres = 0
            blobs, numB = measure.label(orig > thres, return_num=True)
            nPixels = np.zeros(numB + 1)
            x, y, count = 0, 0, 0
            while orig[x, y] != 0:
                if count % 2 == 0:
                    x += 1
                else:
                    y += 1
                count += 1
                if x == width or y == height:
                    x, y = 0, 0
                    break
            zeroth = blobs[x, y]
            for i in range(numB + 1):
                if i != zeroth:
                    nPixels[i] = sum(sum(blobs == i))
            idx = np.argmax(nPixels)
            part_external = blobs == idx
            hold_outside = measure.label(part_external == 0)

            lt, rt, lb, rb = hold_outside[0,0], hold_outside[0,height-1], hold_outside[width-1,0], hold_outside[width-1,height-1]
            outside_external = np.zeros_like(hold_outside)
            for i in range(width):
                for j in range(width):
                    if hold_outside[i,j] in [lt,rt,lb,rb]:
                        outside_external[i,j]=1
            external = outside_external == 0
            orig, recon = orig[external == 1].astype(int), recon[external == 1].astype(int)
        if args.region=="full" or args.region=="body":
            # Evaluate reconstructions.
            if args.distance=="MSE":
                dist = metrics.mean_squared_error(orig,recon)
            elif args.distance=="WD":
                orig = orig.flatten()
                recon = recon.flatten()
                dist = stats.wasserstein_distance(orig,recon)
            dists['filename'].append(fname)
            dists['dist'].append(dist)
        else:
            max_metric, r, c = 0, 0, 0
            outfile = os.path.join(f'{args.patch_path}',fname)
            if not os.path.exists(outfile):
                for row in range(0, orig.shape[0]-args.patch_size):
                    for col in range(0, orig.shape[0]-args.patch_size):
                        orig_patch = orig[row:row+args.patch_size, col:col+args.patch_size]
                        recon_patch = recon[row:row+args.patch_size, col:col+args.patch_size]
                        if args.distance=="MSE":
                            dist = metrics.mean_squared_error(orig_patch,recon_patch)
                        elif args.distance=="WD":
                            orig_patch = orig_patch.flatten()
                            recon_patch = recon_patch.flatten()
                            dist = stats.wasserstein_distance(orig_patch,recon_patch)
                        if dist > max_metric:
                            max_metric = dist
                            r, c = row, col
                dists['filename'].append(fname)
                dists['dist'].append(max_metric)
                orig_recon = np.zeros((args.patch_size,args.patch_size*2))
                orig_recon[:,:args.patch_size] = orig[r:r+args.patch_size,c:c+args.patch_size]
                orig_recon[:,args.patch_size:] = recon[r:r+args.patch_size,c:c+args.patch_size]
                if args.patch_path:
                    imageio.imwrite(outfile,orig_recon.astype(np.uint8))
                    
    # Write out distances to CSV.
    dist_df = pd.DataFrame(dists)
    dist_df.to_csv(args.out_path, index=False)
