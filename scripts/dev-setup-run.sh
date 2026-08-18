# This script assumes that:
#   - The location of the repository is `~/backlooper_backend`
#   - The virtual environment is located at `~/backlooper_backend/.venv`
#   - The input and output devices are both at index 0
#   - I2C is enabled, LCD contrast dialed properly
#   - TODO: MIDI

export INPUT_DEVICE_ID=0
export OUTPUT_DEVICE_ID=0
# TODO: MIDI

ulimit -n 1048576  # workaround for https://github.com/spmvg/backlooper_backend/issues/3

cd ~/backlooper_backend
github_reachable=false
for attempt in 1 2 3; do
	echo "Checking GitHub connectivity (attempt $attempt/3, timeout 2 s)"
	if timeout 2 bash -c '</dev/tcp/github.com/443' 2>/dev/null; then
		echo "GitHub is reachable"
		github_reachable=true
		break
	fi
	echo "GitHub is not reachable on attempt $attempt/3"
	done

if "$github_reachable"; then
	echo "Pulling latest Backlooper version"
	timeout 10 git pull || echo "git pull failed, continuing with local version"
else
	echo "GitHub unavailable after 3 attempts; continuing with local version"
fi
source .venv/bin/activate
python -m pip install -e .
python -m backlooper
