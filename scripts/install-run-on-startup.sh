#!/bin/bash
# Run once to register the systemd system service. Re-run after changing this file.
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
BACKLOOPER_SCRIPT="${SCRIPT_DIR}/dev-setup-run.sh"

sudo tee /etc/systemd/system/backlooper.service > /dev/null <<EOF
[Unit]
Description=Backlooper
After=sound.target network.target

[Service]
ExecStart=/bin/bash ${BACKLOOPER_SCRIPT}
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable backlooper.service
echo "Service registered. Logs after reboot: journalctl -u backlooper -f"
