## Run Raft-Stereo

[Download models](https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip&dl=1) and place into `models/` folder (or use official doc method - you can find on [official page](https://github.com/princeton-vl/RAFT-Stereo)).

Raft-Stereo is a Python-based codebase, so we will set up a dedicated environment for it. The GPU is required, and so `conda` will be used. I already have the latest SDK installed and will use a matching Torch versions for it:

```bash
conda create -n raftstereo124
conda activate raftstereo124
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
