#!/bin/bash
# Run once to register the systemd user service. Re-run after changing backlooper.service.
set -e

mkdir -p ~/.config/systemd/user
cp ~/backlooper_backend/scripts/backlooper.service ~/.config/systemd/user/backlooper.service
systemctl --user daemon-reload
systemctl --user enable --now backlooper.service
echo "Service installed and started. Logs: journalctl --user -u backlooper -f"
