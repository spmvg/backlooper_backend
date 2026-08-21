#!/bin/bash
# Run once to register the systemd system service. Re-run after changing backlooper.service.
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export BACKLOOPER_SCRIPT="${SCRIPT_DIR}/dev-setup-run.sh"

envsubst < "${SCRIPT_DIR}/backlooper.service" | sudo tee /etc/systemd/system/backlooper.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now backlooper.service
echo "Service installed and started. Logs: journalctl -u backlooper -f"
