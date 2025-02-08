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
python RAFT-Stereo/demo.py --restore_ckpt models/raftstereo-realtime.pth --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision
 --left=test/left.png --right=test/right.png --output=result.png
```