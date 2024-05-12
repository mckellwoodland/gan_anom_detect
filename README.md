# gan_anom_detect
Official Repository for the paper "Generative Modeling for Anomaly Detection in Radiological Applications"

# Dependencies

This code was built with Python 3 with package versions listed in the `requirements.txt` file. We also provided a Docker container that can be built by running the following command in this directory.
```
docker build --tag gan_anom_detect .
```
The container can be run interactively as follows, with the current directory being mounted:
```
docker run -it --rm -v $(pwd):/workspace gan_anom_detect /bin/bash
```

# Reconstruct images with StyleGAN2-ADA

Build the Docker container.
```
docker build --tag sg2ada:latest stylegan2-ada-pytorch/.
```

Reconstruct images.
```
stylegan2-ada-pytorch/docker_run.sh python stylegan2-ada-pytorch/projector.py \
						--network {MODEL_PKL} \
						--target {INPUT_DIR} \
						--save-video False \
						--outdir {OUTPUT_DIR}
```

# Evaluate reconstructions

Evaluate reconstructions patch-wise. Corresponding original and reconstructed images must have the same name. Code is built for 2-dimensional grayscale PNG images.
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
