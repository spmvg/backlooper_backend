# This script assumes that:
#   - The location of the repository is `~/backlooper_backend`
#   - The virtual environment is located at `~/backlooper_backend/.venv`
#   - The input and output devices are both at index 0

export INPUT_DEVICE_ID=0
export OUTPUT_DEVICE_ID=0

ulimit -n 1048576  # workaround for https://github.com/spmvg/backlooper_backend/issues/3

cd ~/backlooper_backend
git pull
source .venv/bin/activate
python -m pip install -e .
python -m backlooper
