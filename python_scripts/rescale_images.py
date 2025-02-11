"""
Resize images in a directory.
"""
# Imports.
import argparse
import os
import tqdm
from PIL import Image

# Arguments.
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-i','--in_dir',type=str,required=True,help='Path to the directory containing images to be resized.')
required.add_argument('-o','--out_dir',type=str,required=True,help='Path to the directory to put resized images into.')

optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-r','--resolution',type=int,default=512,help='Desired output resolution.\
                                                                     Defaults to 512.')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

# Main code.
if __name__=="__main__":
    for filename in tqdm.tqdm(os.listdir(args.in_dir)):
        out_file = os.path.join(args.out_dir, filename)
        if not os.path.exists(out_file):
            image = Image.open(os.path.join(args.in_dir, filename))
            new_image = image.resize((args.resolution, args.resolution))
            new_image.save(out_file)
