#!/usr/bin/env bash
# Run in an interactive terminal: installing system packages requires sudo.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
rules_dir="$repo_root/.uv/realsense"
mkdir -p "$rules_dir"
rules="$rules_dir/99-realsense-libusb.rules"
curl --fail --location --proto '=https' --tlsv1.2 \
  https://raw.githubusercontent.com/realsenseai/librealsense/v2.58.4/config/99-realsense-libusb.rules \
  --output "$rules"
udevadm verify "$rules"
sudo -v
if ! command -v candump >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y can-utils
fi
sudo modprobe gs_usb
sudo install -m 644 "$rules" /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules
echo "CAN utilities/driver and RealSense rules installed. Reconnect the cameras."
echo "The four CAN adapters still need a verified role-to-USB mapping."
