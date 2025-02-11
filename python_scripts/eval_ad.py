"""
Evaluate AD detection via AUROC when given reconstruction distances.
"""

# Imports.
import argparse
import numpy as np
import pandas as pd
import random
import tqdm
from sklearn import metrics

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-a','--anomaly',type=str,required=True,help='Path to csv file containing the distances for the anomalous dataset.')
required.add_argument('-b','--baseline',type=str,required=True,help='Path to csv file containing the distances for the baseline data.')
required.add_argument('-o','--out_path',type=str,required=True,help='Path to csv file to write AUROCS to.')

optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-n','--num_samp',type=int,default=50,help='The number of bootstrap samples to use.')

args = parser.parse_args()

# Main Code.
if __name__=="__main__":
    # Read in distances
    base_df = pd.read_csv(args.baseline)
    anom_df = pd.read_csv(args.anomaly)
    base_dist = list(base_df['dist'])
    anom_dist = list(anom_df['dist'])

    # Iterate through bootstrap samples.
    base_len = len(base_dist)
    anom_len = len(anom_dist)
    min_samp = min(base_len, anom_len)
    aucs = []
    truth = [0] * min_samp + [1] * min_samp
    for _ in tqdm.tqdm(range(args.num_samp)):
        base_sam = random.sample(base_dist,min_samp)
        anom_sam = random.sample(anom_dist,min_samp)
        n_max = max(max(base_sam),max(anom_sam))

        # Calculate AUROC.
        pred = np.array(base_sam + anom_sam) / n_max
        fpr, tpr, thresholds = metrics.roc_curve(truth, pred)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
    print(np.mean(aucs),np.std(aucs))
    out_csv = pd.DataFrame({'AUROC':aucs})
    out_csv.to_csv(args.out_path)
