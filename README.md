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

# Evaluate reconstructions

Evaluate reconstructions patch-wise. Corresponding original and reconstructed images must have the same name. Code is built for 2-dimensional PNG images.
```
usage: eval_recon_patch.py [-h] [-d DISTANCE]

Optional Arguments:
  -d DISTANCE, --distance DISTANCE
                        One of [MSE, WD, SS] for mean-squared error, Wasserstein distance, or Structural Similarity.
                        Defaults to MSE.
```
