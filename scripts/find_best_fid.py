"""
Locates the associated pickle file with the lowest associated fid50k_full.
"""
# Imports.
import argparse
import json
import pandas as pd

# Arguments.
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
required.add_argument('-f',
                      '--fname',
                      type=str,
                      help='Path to the StyleGAN2-ADA output JSON file with metric information, i.e. the "metric-fid50k_full" JSON file.')
args = parser.parse_args()

# Main code.
if __name__=="__main__":
    with open(args.fname) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    idx = df['results.fid50k_full'].argmin()
    results = df.iloc[idx]
    print("Network PKL", results['snapshot_pkl'])
    print("FID", results['results.fid50k_full'])
