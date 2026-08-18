# This script assumes that:
#   - The location of the repository is `~/backlooper_backend`
#   - The virtual environment is located at `~/backlooper_backend/.venv`
#   - The input and output devices are both at index 0
#   - I2C is enabled, LCD contrast dialed properly

export INPUT_DEVICE_ID=0
export OUTPUT_DEVICE_ID=0
export MIDI_PORT_NAME='MPK Mini Mk II:MPK Mini Mk II MIDI 1 28:0'

ulimit -n 1048576  # workaround for https://github.com/spmvg/backlooper_backend/issues/3

cd ~/backlooper_backend
github_reachable=false
for attempt in {1..3}; do
	echo "Checking GitHub DNS (attempt $attempt/12)"
	if getent hosts github.com >/dev/null; then
		echo "GitHub DNS is available"
		github_reachable=true
		break
	fi

	if [ "$attempt" -lt 3 ]; then
		echo "GitHub DNS unavailable; retrying in 5 s"
		sleep 5
	fi
done

if "$github_reachable"; then
	echo "Pulling latest Backlooper version"
	timeout 10 git pull || echo "git pull failed, continuing with local version"
else
	echo "GitHub DNS unavailable after 12 attempts; continuing with local version"
fi
source .venv/bin/activate
python -m pip install -e .
python -m backlooper
