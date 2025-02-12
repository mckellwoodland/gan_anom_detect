"""
Script to convert DICOM to NIfTI images.
"""

# Imports.
import argparse
import dicom2nifti
import os
import pydicom
import tqdm
import numpy as np

# Arguments.
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-i','--in_dir',type=str,required=True,help='Path to the directory containing DICOMs to be converted.')
required.add_argument('-o','--out_dir',type=str,required=True,help='Path to the directory to put the NIfTIs.')

args = parser.parse_args()

# Main script.
if __name__ == "__main__":
    # Convert DICOM directories to NIfTI directories.
    for folder in tqdm.tqdm(os.listdir(args.in_dir)):
        dcm_path = os.path.join(args.in_dir, folder)
        out_path = os.path.join(args.out_dir, folder)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        dicom2nifti.convert_directory(dcm_path, out_path)
    # Convert NIfTI directories to a single directory.
    for folder in tqdm.tqdm(os.listdir(args.out_dir)):
        for ni_file in os.listdir(os.path.join(args.out_dir, folder)):
            os.rename(os.path.join(args.out_dir, folder, ni_file), os.path.join(args.out_dir, folder) + '.nii.gz')
        os.rmdir(os.path.join(args.out_dir, folder))
