"""
Window a computed tomography image in NIfTI format.
"""

# Imports.
import argparse
import os
import tqdm
import numpy as np
import nibabel as nib

# Arguments.
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-i','--in_dir',type=str,required=True,help='Path to the directory containing images to be windowed. \
                                                                  Images must be in NIfTI format.')
required.add_argument('-o','--out_dir',type=str,required=True,help='Path to the directory to put the windowed images into.')

optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-w','--window_width',type=float,default=350.,help='Window width.')
optional.add_argument('-l','--window_level',type=float,default=50.,help='Window level.')

args = parser.parse_args()

# Main code.
if __name__=="__main__":
    for filename in tqdm.tqdm(os.listdir(args.in_dir)):
        out_path = os.path.join(args.out_dir, filename)
        if not os.path.exists(out_path):
            # Load in image.
            path = os.path.join(args.in_dir, filename)
            img = nib.load(path)
            img_data = img.get_fdata()

            # Window the image.
            img_data = np.clip(img_data, args.window_level - (args.window_width/2.), args.window_level + (args.window_width/2.))

            # Save image.
            img_new = nib.Nifti1Image(img_data, img.affine, img.header)
            nib.save(img_new, out_path)
