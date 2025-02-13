"""
Convert NIfTI slices to PNG.
"""
# Inputs.
import argparse
import imageio
import os
import tqdm
import nibabel as nib
import numpy as np

# Arguments.
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-i','--in_dir',type=str,required=True,help='Path to the directory containing NIfTI slices to be converted.')
required.add_argument('-o','--out_dir',type=str,required=True,help='Path to the directory to put the PNG images.')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

# Functions.
def rescale255(img):
    """
    Rescale image data to be in the range 0-255 of unsigned integer type.

    Input:
        img(ndarray): Image data to be rescaled.

    Output:
        img(ndarray): Rescaled image data.
    """
    min_val = img.min()
    if min_val < 0:
        img += abs(min_val)
    else:
        img -= min_val
    img /= img.max()
    img = img * 255.
    img = img.astype(np.uint8)
    return img

# Main script.
if __name__ == "__main__":
    # Iterate through all slices in a directory.
    for filename in tqdm.tqdm(os.listdir(args.in_dir)):
        # Load in the slices.
        img = nib.load(os.path.join(args.in_dir, filename))
        img_data = img.get_fdata()

        # Properly reorient image.
        img_data = np.flip(np.rot90(np.rot90(np.rot90(img_data))), axis=1)

        # Prepare data for PNG conversion.
        img_data = rescale255(img_data)

        # Save as PNG image.
        out_path = os.path.join(args.out_dir, filename[:-6]+'png')
        imageio.imwrite(out_path, img_data)
