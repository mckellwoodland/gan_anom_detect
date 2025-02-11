"""
Adds either Gaussian noise or blur to a dataset of grayscale images to bring context to the Frechet distance calculations.
"""
# Imports.
import argparse
import cv2
import numpy as np
import os
import tqdm
from PIL import Image

# Arguments.
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-i','--in_dir',type=str,required=True,help='Path to directory containing images to be manipulated.')
required.add_argument('-o','--out_dir',type=str,required=True,help='Path to directory to put the manipulated images into.')

optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-k','--kernel_size',type=str,default="(5,5)",help='Size of Gaussian kernel (blur).\
                                                                          Defaults to (5,5).')
optional.add_argument('-m','--mean',type=float,default=0.,help='Mean of Gaussian distribution (noise).\
                                                                Defaults to 0.')
optional.add_argument('-s','--sigma',type=float,default=0.,help='Standard deviation of kernel.\
                                                                 Defaults to 0.')
optional.add_argument('-t','--type',type=str,default='n',help='Type of image manipulation: Gaussian noise (n) or blur (b).\
                                                               Defaults to n.')
optional.add_argument('-v','--var',type=float,default=0.01,help='Variance of Gaussian distribution (noise).\
                                                                 Defaults to 0.01.')
args = parser.parse_args()

# Functions.
def add_gaussian_blur(img,kernel_size,sigma):
    """
    Add Gaussian blur to an image.
    Inputs:
        - img: Image to be manipulated.
        - kernel_size: Size of Gaussian kernel.
                       Must be odd and positive.
        - sigma: Standard deviation of kernel.
                 Calculated with kernel size if 0.
    Returns:
        Image with added blur.
    """
    return cv2.GaussianBlur(img,kernel_size,sigma)

def add_gaussian_noise(img,mean,var):
    """
    Add Gaussian noise to an image.
    Inputs:
        - img: Image to be manipulated.
        - mean: Mean of Gaussian distribution.
        - var: Variance of Gaussian distribution.
    Returns:
        Image with added noise.
    """
    sigma = var**0.5
    gaussian = np.random.normal(mean,sigma,img.shape)
    return np.clip(img+gaussian,0,1)

# Main code.
if __name__=="__main__":
    for img_name in tqdm.tqdm(os.listdir(args.in_dir)):
        # Read in image.
        img_pth = os.path.join(args.in_dir,img_name)
        img = Image.open(img_pth)
        # Convert to grayscale.
        img = img.convert('L')
        # Convert to NumPy.
        img = np.array(img).astype(float)
        # Convert range to 0-1.
        img /= 255.
        # Add Gaussian noise.
        if args.type == 'n':
            img_m = add_gaussian_noise(img,args.mean,args.var)
        elif args.type == 'b':
            img_m = add_gaussian_blur(img,eval(args.kernel_size),args.sigma)
        # Convert to grayscale PIL image.
        img_m = (img_m * 255).astype(np.uint8)
        img_m = Image.fromarray(img_m)
        # Write manipulated image out to new location.
        out_pth = os.path.join(args.out_dir,img_name)
        img_m.save(out_pth)
