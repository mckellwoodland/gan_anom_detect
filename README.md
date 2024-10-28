# gan_anom_detect
Official Repository for the paper "Generative Modeling for Anomaly Detection in Radiological Applications"

# Dependencies

This code was built with Python 3 with package versions listed in the `requirements.txt` file.
We also provide a Docker container that can be built by running the following command in this directory.
```
docker build --tag gan_anom_detect .
```

Our Python scripts can be run directly after entering the Docker container
```
docker run -it --rm -v $(pwd):/workspace gan_anom_detect /bin/bash
```
and providing the appropriate Python commands arguments or by editing the bash scripts with your arguments. 
The bash script will run the container and provide the arguments to the script for you. 
You will need to edit the above command or the bash scripts if you want a directory other than the current directory mounted.

To use the StyleGAN2-ADA<sup>1</sup> fork submodule, you'll need the following container:
```
docker build --tag sg2ada:latest stylegan2-ada-pytorch/.
```

To use the StudioGAN<sup>2</sup> fork submodule, you can pull the following container:
```
docker pull alex4727/experiment:pytorch113_cuda116
```
To use our bash scripts, you'll need the following Docker container which updated the PyTorch version:
```
docker build --tag studiogan:latest PyTorch-StudioGAN/.
```

Lastly, the following Docker container is compatible with the `frd-score` Python package<sup>3</sup>.
```
docker build --tag frd:latest frd/.
```

# Train a StyleGAN2-ADA model.

Train a StyleGAN2-ADA model with the StyleGAN2-ADA repository (forked) by providing the `--outdir` and `--data` arguments to the `train.sh` script.
The provided `Optional Arguments` in the script were the hyperparameters used to train models in our study.
While the official repository and this study works with PNGs, you can train on NiFTI files using the `nifti` branch of our fork.

```
./bash_scripts/train.sh
```
```
Usage: train.py [OPTIONS]

  Train a GAN using the techniques described in the paper "Training
  Generative Adversarial Networks with Limited Data".

  Examples:

  # Train with custom dataset using 1 GPU.
  python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1

  # Train class-conditional CIFAR-10 using 2 GPUs.
  python train.py --outdir=~/training-runs --data=~/datasets/cifar10.zip \
      --gpus=2 --cfg=cifar --cond=1

  # Transfer learn MetFaces from FFHQ using 4 GPUs.
  python train.py --outdir=~/training-runs --data=~/datasets/metfaces.zip \
      --gpus=4 --cfg=paper1024 --mirror=1 --resume=ffhq1024 --snap=10

  # Reproduce original StyleGAN2 config F.
  python train.py --outdir=~/training-runs --data=~/datasets/ffhq.zip \
      --gpus=8 --cfg=stylegan2 --mirror=1 --aug=noaug

  Base configs (--cfg):
    auto       Automatically select reasonable defaults based on resolution
               and GPU count. Good starting point for new datasets.
    stylegan2  Reproduce results for StyleGAN2 config F at 1024x1024.
    paper256   Reproduce results for FFHQ and LSUN Cat at 256x256.
    paper512   Reproduce results for BreCaHAD and AFHQ at 512x512.
    paper1024  Reproduce results for MetFaces at 1024x1024.
    cifar      Reproduce results for CIFAR-10 at 32x32.

  Transfer learning source networks (--resume):
    ffhq256        FFHQ trained at 256x256 resolution.
    ffhq512        FFHQ trained at 512x512 resolution.
    ffhq1024       FFHQ trained at 1024x1024 resolution.
    celebahq256    CelebA-HQ trained at 256x256 resolution.
    lsundog256     LSUN Dog trained at 256x256 resolution.
    <PATH or URL>  Custom network pickle.

Options:
  --outdir DIR                    Where to save the results  [required]
  --gpus INT                      Number of GPUs to use [default: 1]
  --snap INT                      Snapshot interval [default: 50 ticks]
  --metrics LIST                  Comma-separated list or "none" [default:
                                  fid50k_full]

  --seed INT                      Random seed [default: 0]
  -n, --dry-run                   Print training options and exit
  --data PATH                     Training data (directory or zip)  [required]
  --cond BOOL                     Train conditional model based on dataset
                                  labels [default: false]

  --subset INT                    Train with only N images [default: all]
  --mirror BOOL                   Enable dataset x-flips [default: false]
  --cfg [auto|stylegan2|paper256|paper512|paper1024|cifar]
                                  Base config [default: auto]
  --gamma FLOAT                   Override R1 gamma
  --kimg INT                      Override training duration
  --batch INT                     Override batch size
  --aug [noaug|ada|fixed]         Augmentation mode [default: ada]
  --p FLOAT                       Augmentation probability for --aug=fixed
  --target FLOAT                  ADA target value for --aug=ada
  --augpipe [blit|geom|color|filter|noise|cutout|bg|bgc|bgcf|bgcfn|bgcfnc]
                                  Augmentation pipeline [default: bgc]
  --resume PKL                    Resume training [default: noresume]
  --freezed INT                   Freeze-D [default: 0 layers]
  --fp32 BOOL                     Disable mixed-precision training
  --nhwc BOOL                     Use NHWC memory format with FP16
  --nobench BOOL                  Disable cuDNN benchmarking
  --allow-tf32 BOOL               Allow PyTorch to use TF32 internally
  --workers INT                   Override number of DataLoader workers
  --beta0 FLOAT                   Beta_0
  --help                          Show this message and exit.
```

# Evaluate StyleGAN2's Generative Quality

Generate 50,000 images using the model weights associated with the lowest Fréchet Inception Distance (FID) attained during training. 
Images can be generated by providing `--network` and `--outdir` arguments to `generator.sh`.
```
./bash_scripts/generator.sh
```
```
Usage: generate.py [OPTIONS]

  Generate images using pretrained network pickle.

  Examples:

  # Generate curated MetFaces images without truncation (Fig.10 left)
  python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

  # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
  python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

  # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
  python generate.py --outdir=out --seeds=0-35 --class=1 \
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

  # Render an image from projected W
  python generate.py --outdir=out --projected_w=projected_w.npz \
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

Options:
  --network TEXT                  Network pickle filename  [required]
  --seeds NUM_RANGE               List of random seeds
  --trunc FLOAT                   Truncation psi  [default: 1]
  --class INTEGER                 Class label (unconditional if not specified)
  --noise-mode [const|random|none]
                                  Noise mode  [default: const]
  --projected-w FILE              Projection result file
  --outdir DIR                    Where to save the output images  [required]
  --help                          Show this message and exit.
```

`find_best_fid.sh` given the `--fname` argument can be used to determine which weights were associated with the lowest FID score.
```
./bash_scripts/find_best_fid.sh
```
```
usage: find_best_fid.py [-h] [-f FNAME]

Required Arguments:
  -f FNAME, --fname FNAME
                        Path to the StyleGAN2-ADA output JSON file with metric information, i.e.
                        the "metric-fid50k_full" JSON file.
```

Evaluate FID and Fréchet SwAV Distance (FSD) with the StudioGAN<sup>2</sup> by providing `--dset1`, `--dset2`, `--eval_backbone` (either `InceptionV3_torch`<sup>4</sup> or SwAV `SwAV_torch`<sup>5</sup>), and `--out_path` to the `eval_fd.sh` script. The batch size `--batch_size` argument can also be updated if memory issues are encountered.

When evaluating a generative distribution, the first dataset consists of real images (training images), and the second consists of the generated images (or vice versa). 
When determining a baseline, the first and second datasets come from a random split of the real images. 
In both cases, the folder name containing both image datasets must match (such as `class0`).

```
./bash_scripts/eval_fd.sh
```
```
usage: evaluate.py [-h] [-metrics EVAL_METRICS [EVAL_METRICS ...]] [--post_resizer POST_RESIZER] [--eval_backbone EVAL_BACKBONE] [--dset1 DSET1]
                   [--dset1_feats DSET1_FEATS] [--dset1_moments DSET1_MOMENTS] [--dset2 DSET2] [--batch_size BATCH_SIZE] [--seed SEED] [-DDP]
                   [--backend BACKEND] [-tn TOTAL_NODES] [-cn CURRENT_NODE] [--num_workers NUM_WORKERS] --out_path OUT_PATH

optional arguments:
  -h, --help            show this help message and exit
  -metrics EVAL_METRICS [EVAL_METRICS ...], --eval_metrics EVAL_METRICS [EVAL_METRICS ...]
                        evaluation metrics to use during training, a subset list of ['fid', 'is', 'prdc'] or none
  --post_resizer POST_RESIZER
                        which resizer will you use to evaluate GANs in ['legacy', 'clean', 'friendly']
  --eval_backbone EVAL_BACKBONE
                        [InceptionV3_tf, InceptionV3_torch, ResNet50_torch, SwAV_torch, DINO_torch, Swin-T_torch]
  --dset1 DSET1         specify the directory of the folder that contains dset1 images (real).
  --dset1_feats DSET1_FEATS
                        specify the path of *.npy that contains features of dset1 (real). If not specified, StudioGAN will automatically extract
                        feat1 using the whole dset1.
  --dset1_moments DSET1_MOMENTS
                        specify the path of *.npy that contains moments (mu, sigma) of dset1 (real). If not specified, StudioGAN will
                        automatically extract moments using the whole dset1.
  --dset2 DSET2         specify the directory of the folder that contains dset2 images (fake).
  --batch_size BATCH_SIZE
                        batch_size for evaluation
  --seed SEED           seed for generating random numbers
  -DDP, --distributed_data_parallel
  --backend BACKEND     cuda backend for DDP training \in ['nccl', 'gloo']
  -tn TOTAL_NODES, --total_nodes TOTAL_NODES
                        total number of nodes for training
  -cn CURRENT_NODE, --current_node CURRENT_NODE
                        rank of the current node
  --num_workers NUM_WORKERS
  --out_path OUT_PATH   output file to put metrics into
```

The Fréchet Radiomics Distance (FRD)<sup>3</sup> can be calculated by providing the paths to the two datasets to `eval_frd.sh`.
```
./bash_scripts/eval_frd.sh
```

# Reconstruct images with StyleGAN2-ADA

This code uses a fork of the official StyleGAN2-ADA repository to reconstruct the images with a trained StyleGAN2-ADA model via backpropagation. You'll need to be on the `proj_dir` branch of the repository to use our expanded capabilities of reconstructing all images in a given directory. To reconstruct the images, provide `--network`, `--target`, and `--outdir` to the `projector.sh` script.

```
./bash_scripts/projector.sh
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

# References
1. Tero Karras *et al.* Training generative adversarial networks with limited data. In NeurIPS 2020; Curran Associates, Inc; 33:12104-12114; 2020.
2. Minguk Kang *et al.* StudioGAN: A taxonomy and benchmark of GANs for image synthesis. TPAMI, 45(12):15725-15742.
3. Osuala *et al.* Towards learning contrast kinetics with multi-condition latent diffusion models. In MICCAI 2024; Springer, Cham; 15005:713-723; 2024.
4. Christian Szegedy *et al.* Going deeper with convolutions. In CVPR 2015, IEEE, pp. 1-9, 2015.
5. Mathilde Caron *et al.* Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS 2020; Curran Associates, Inc.; 33:9912-9924; 2020.
