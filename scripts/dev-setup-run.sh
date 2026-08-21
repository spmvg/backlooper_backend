# This script assumes that:
#   - The repository contains this script under scripts/
#   - The virtual environment is located at <repo>/\.venv
#   - The input and output devices are both at index 0
#   - I2C is enabled, LCD contrast dialed properly

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.."; pwd)"

export INPUT_DEVICE_ID=0
export OUTPUT_DEVICE_ID=0
export MIDI_PORT_MATCH='MPK Mini Mk II'

ulimit -n 1048576  # workaround for https://github.com/spmvg/backlooper_backend/issues/3

# Prevent USB audio device from suspending mid-session (causes callback stalls and silent loops).
echo -1 | tee /sys/module/usbcore/parameters/autosuspend > /dev/null

cd "$REPO_DIR"
github_reachable=false
for attempt in {1..3}; do
	echo "Checking GitHub DNS (attempt $attempt/3)"
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
	echo "GitHub DNS unavailable after 3 attempts; continuing with local version"
fi
source "$REPO_DIR/.venv/bin/activate"
python -m pip install -e .
python -m backlooper
