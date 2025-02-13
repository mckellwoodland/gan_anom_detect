"""
Convert a 3D NIfTI image into 2D NIfTI slices.
"""

# Imports.
import argparse
import os
import tqdm
import nibabel as nib
import numpy as np

# Arguments.
parser = argparse.ArgumentParser()
parser._action_groups.pop()

def none_or_str(value):
    if value == "None":
        return None
    return value

required = parser.add_argument_group('Required Arguments')
required.add_argument('-i','--in_dir',type=str,required=True,help='Path to the directory containing the 3D NIfTI images to be sliced.')
required.add_argument('-o','--out_dir',type=str,required=True,help='Path to the directory to put the 2D NIfTI slices.')

optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-m','--mask_dir',type=none_or_str,default=None,help='Path to the directory containing masks.\
                                                                            The masks must have the same filename as the original image.\
                                                                            The mask filename must not contain a period.\
                                                                            The slices containing the mask must be consecutive. \
                                                                            All saved slices will contain the given structure.\
                                                                            If None, all slices will be saved.\
                                                                            Default is None.')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

# Main Script.
if __name__=="__main__":
    # Iterate through 3D NIfTI files.
    for filename in tqdm.tqdm(os.listdir(args.in_dir)):
        prefix = filename.split('.')[0]

        # Read in image.
        img = nib.load(os.path.join(args.in_dir, filename))
        img_data = img.get_fdata()

        # Only keep the slices that pertain to the mask.
        first_s, last_s = 0, img_data.shape[2]
        if args.mask_dir:
            mask = nib.load(os.path.join(args.mask_dir, filename))
            mask_data = mask.get_fdata()
            first = False
            for i in range(mask_data.shape[2]):
                if np.sum(mask_data[:, :, i]) != 0.0 and first is False:
                    first_s = i
                    first = True
                elif np.sum(mask_data[:, :, i]) == 0.0 and first is True:
                    last_s = i
                    break
            img_data = img_data[:, :, first_s:last_s]
        
        # Write out individual slices.
        for i in range(img_data.shape[2]):
            img_slice = img_data[:, :, i]
            out_pth = os.path.join(args.out_dir, prefix + f"_{i+first_s}.nii.gz")
            out_img = nib.Nifti1Image(img_slice, img.affine, img.header)
            nib.save(out_img, out_pth)
