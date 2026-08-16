# This script assumes that:
#   - The location of the repository is `~/backlooper_backend`
#   - The virtual environment is located at `~/backlooper_backend/.venv`

cd ~/backlooper_backend
source .venv/bin/activate
python -m pip install -e .
python -m backlooper
