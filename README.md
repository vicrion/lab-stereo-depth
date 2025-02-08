## Run Raft-Stereo

Raft-Stereo is a Python-based codebase, so we will set up a dedicated environment for it.

Download models (or use official doc method - either manual or through a suggested `bash` script):
```bash
make models.fetch
make models.install
```

New local environment and dependencies:
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```