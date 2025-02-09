## Run RAFT-Stereo

[Download models](https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip&dl=1) and place into `models/` folder (or use official doc method - you can find on [official page](https://github.com/princeton-vl/RAFT-Stereo)).

RAFT-Stereo is a Python-based codebase, so we will set up a dedicated environment for it. The GPU usage is assumed, and so `conda` will be used. I already have the latest CUDA SDK installed (version 12.4) and will use a matching Torch versions for it:

```bash
conda create -n raftstereo124
conda activate raftstereo124
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# other reqs:
pip3 install matplotlib tensorboard scipy opencv-python tqdm opt_einsum imageio scikit-image
conda install p7zip
```

Run a test to verify the codebase works within the established `conda` environment:

```bash
cd RAFT-Stereo
# If on Windows, we want to reverse the '\' to '/' in the full paths for left and right!
# make sure to use the correct full path for all the data and models:
python demo.py --restore_ckpt C:/Users/nordw/github/SM-stereo-depth/models/iraftstereo_rvc.pth --context_norm instance -l=C:/Users/nordw/github/SM-stereo-depth/test/left/*.png -r=C:/Users/nordw/github/SM-stereo-depth/test/right/*.png --output_directory C:/Users/nordw/github/SM-stereo-depth/test/out-rvc
```

We can try a different model as well, for example, to optmize for speed. The faster model requires an additional build step; I had to move all the CUDA code to the parent directory as a few fixes were necessary for more recent CUDA SDK. To build and run:
```bash
# build optimization code, it will use CUDA based optimizations directly:
pip3 install RAFT-Stereo-sampler/.
# run optimized model:
python demo.py --restore_ckpt C:/Users/nordw/github/SM-stereo-depth/models/raftstereo-realtime.pth --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision -l=C:/Users/nordw/github/SM-stereo-depth/test/left/*.png -r=C:/Users/nordw/github/SM-stereo-depth/test/right/*.png --output_directory C:/Users/nordw/github/SM-stereo-depth/test/out-realtime
```