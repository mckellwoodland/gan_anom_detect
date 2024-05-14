# gan_anom_detect
Official Repository for the paper "Generative Modeling for Anomaly Detection in Radiological Applications"

# Dependencies

This code was built with Python 3 with package versions listed in the `requirements.txt` file. We also provided a Docker container that can be built by running the following command in this directory.
```
docker build --tag gan_anom_detect .
```

To reconstruct images using the StyleGAN2-ADA code, you'll also need the following container.
```
docker build --tag sg2ada:latest stylegan2-ada-pytorch/.
```

Our Python scripts can either be run directly after entering the Docker container by providing the appropriate arguments
```
docker run -it --rm -v $(pwd):/workspace gan_anom_detect /bin/bash
```
or by editing the bash scripts with your arguments. The bash script will run the container and provide the arguments to the script for you. You will need to edit the above command or the bash scripts if you want a directory other than the current directory mounted.

# Train a StyleGAN2-ADA model.

You can train a StyleGAN2-ADA model with the official StyleGAN2-ADA repository (forked). While the repository and this study works with PNGs, you can train on NiFTI files using the `nifti` branch of our fork.

```
stylegan2-ada-pytorch/docker_run.sh python stylegan2-ada-pytorch/train.py --outdir {OUT_DIR} \
                                                                          --gpus {GPUS} \
                                                                          --data {DATA_DIR} \
                                                                          --cfg stylegan2 \
                                                                          --augpipe bgcfnc \
                                                                          --gamma 8.2
```

# Reconstruct images with StyleGAN2-ADA

This code uses a fork of the official StyleGAN2-ADA repository to reconstruct the images with a trained StyleGAN2-ADA model via backpropagation. You'll need to be on the `proj_dir` branch of the repository to use our expanded capabilities of reconstructing all images in a given directory. 

Provide the path to the model weights `MODEL_PKL`, the path to the directory containing the original images `INPUT_DIR`, and the path to the directory to put the reconstructed images into `OUTPUT_DIR` to the `projector.sh` script.

```
./scripts/projector.sh
```
```
Usage: projector.py [OPTIONS]

  Project given image to the latent space of pretrained network pickle.

  Examples:

  python projector.py --outdir=out --target=~/mytargetimg.png \
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

Options:
  --network TEXT        Network pickle filename  [required]
  --target FILE         Target image file to project to  [required]
  --num-steps INTEGER   Number of optimization steps  [default: 1000]
  --seed INTEGER        Random seed  [default: 303]
  --save-video BOOLEAN  Save an mp4 video of optimization progress  [default:
                        True]
  --outdir DIR          Where to save the output images  [required]
  --help                Show this message and exit.
```

To decide on the model weights to use, we provide a script `find_best_fid.sh` to localize the StyleGAN2-ADA weights associated with the lowest FID score given the path to the StyleGAN2-ADA JSON results file.
```
./scripts/find_best_fid.sh
```
```
usage: find_best_fid.py [-h] [-f FNAME]

Required Arguments:
  -f FNAME, --fname FNAME
                        Path to the StyleGAN2-ADA output JSON file with metric information, i.e.
                        the "metric-fid50k_full" JSON file.
```

# Evaluate reconstructions

This code evaluates reconstructions patch-wise. Corresponding original and reconstructed images must have the same name. Code is built for 2-dimensional grayscale PNG images.

Provide the paths to the directories containing the original `ORIG_DIR` and reconstructed images `RECON_DIR` to the `eval_recon_patch.sh` script. The distance function `DISTANCE` and patch size `PATCH_SIZE` can also be changed from their default values of mean-squared error and 32.
```
./scripts/eval_recon_patch.sh
```
```
usage: eval_recon_patch.py [-h] [-o ORIG_DIR] [-r RECON_DIR] [-d DISTANCE] [-s PATCH_SIZE]

Required Arguments:
  -o ORIG_DIR, --orig_dir ORIG_DIR
                        Path to the directory containing the original images.
  -r RECON_DIR, --recon_dir RECON_DIR
                        Path to the directory containing the reconstructed images.

Optional Arguments:
  -d DISTANCE, --distance DISTANCE
                        One of [MSE, WD, SS] for mean-squared error, Wasserstein distance, or Structural Similarity.
                        Defaults to MSE.
  -s PATCH_SIZE, --patch_size PATCH_SIZE
                        One integer representing the dimensionality of the patch to be evaluated. Defaults to 32.
```
