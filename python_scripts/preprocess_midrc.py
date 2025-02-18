"""
Preproccess the MIDRC data. DICOMs were downsampled to 512x512, with intensities rescaled to [0, 255] and converted to 8-bit unsigned integers, and subsequently converted to PNGs.
"""

# Imports.
import argparse
import cv2
import os
import pydicom
import tqdm
import numpy as np
import SimpleITK as sitk

# Arguments.
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-i','--in_dir',type=str,required=True,help='Path to directory that contains the original DICOMs in subdirectories. The first subset of directories should be named after the body part examined. The second subset should be named after the modality.')
required.add_argument('-o','--out_dir',type=str,required=True,help='Path to directory to put PNGs into.')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

# Functions.
def find_dicom(in_path, anatomy, modality, idx=0):
    """
    Recursively iterate through folders to find DICOMs to be converted.

    Inputs:
        in_path(str): Directory to be searched.
        anom_feat(str): Body part examined.
        modal(str): Study modality.
        idx(int): Index of image in the directory.

    Output:
        idx(int): Index of image in the directory.
    """
    for filename in os.listdir(in_path):
        if '.dcm' in filename:
            dcm = pydicom.dcmread(os.path.join(in_path, filename), force=True)
            img = dcm.pixel_array
            if len(img.shape) > 2:
                for j, img_slice in enumerate(img):
                    write_to_png(img_slice, os.path.join(args.out_dir, f"{anatomy}_{modality}_{idx}_{j}.png"))
            else:
                write_to_png(img, os.path.join(args.out_dir, f"{anatomy}_{modality}_{idx}.png"))
            idx += 1
        else:
            idx = find_dicom(os.path.join(in_path, filename), anatomy, modality, idx)
    return idx

def write_to_png(img, write_path):
    """
    Preprocess a radiograph and write to PNG.

    Inputs:
        img(array): Radiograph to be saved.
        write_path(str): Name of filename to be saved to.
    """
    # Resize image.
    img = cv2.resize(img, (512, 512))
    # Rescale intensities to [0, 255].
    img = sitk.RescaleIntensity(sitk.GetImageFromArray(img))
    # Convert to 8-bit unsigned integers.
    img = sitk.GetArrayFromImage(img).astype(np.uint8)
    # Don't write out completely black slices.
    if img.sum() != 0:
        cv2.imwrite(write_path, img)

# Main script.
if __name__ == "__main__":
    for anom in tqdm.tqdm(os.listdir(args.in_dir)):
        for modal in os.listdir(os.path.join(args.in_dir, anom)):
            find_dicom(args.in_dir, anom, modal)
