#!/usr/bin/env bash
# Provision a fresh Debian Lightsail instance to serve Thumbsmith.
# Idempotent: safe to re-run after a code change or a failed attempt.
#
#   ssh admin@<ip> 'bash -s' < deploy/setup.sh
#
# The API key is NOT handled here — copy .env across separately so the secret never
# passes through a script that lives in git.

set -euo pipefail

REPO="https://github.com/menahilfatimakhan/thumbsmith.git"
APP_DIR="/home/admin/thumbsmith"

echo "==> Ensuring swap exists"
# Lightsail's small instances ship with ~1 GB of RAM and no swap. Decoding frames into
# numpy arrays spikes well past that, and with no swap the kernel OOM-kills gunicorn
# mid-render — which looks like a random 502 rather than a memory problem.
# Read /proc/swaps rather than calling swapon: the swap tools live in /usr/sbin, which is
# not on a non-root PATH on Debian, so `swapon --show` fails with "command not found" and
# the check silently decides there is no swap.
if [ "$(awk 'NR > 1' /proc/swaps | wc -l)" -eq 0 ]; then
    if [ ! -e /swapfile ]; then
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo /usr/sbin/mkswap /swapfile >/dev/null
    fi
    sudo /usr/sbin/swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
    echo "    swap enabled"
else
    echo "    swap already active: $(awk 'NR > 1 {printf "%s (%s KB)", $1, $3}' /proc/swaps)"
fi

echo "==> Installing system packages"
sudo apt-get update -qq
# ffmpeg carries ffprobe, which download.py needs to read the duration of uploads.
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3 python3-venv python3-pip git ffmpeg nginx \
    fonts-dejavu-core libgl1 libglib2.0-0

echo "==> Fetching the code"
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" fetch --quiet origin
    git -C "$APP_DIR" reset --hard --quiet origin/main
else
    git clone --quiet "$REPO" "$APP_DIR"
fi

echo "==> Building the virtualenv"
if [ ! -d "$APP_DIR/.venv" ]; then
    python3 -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
# opencv-python pulls a desktop GUI stack that a headless box has no use for; the
# headless wheel is the same library without it, and much smaller to install.
"$APP_DIR/.venv/bin/pip" install --quiet \
    -r "$APP_DIR/requirements.txt" \
    -r "$APP_DIR/deploy/requirements-server.txt"
"$APP_DIR/.venv/bin/pip" install --quiet --force-reinstall opencv-python-headless

mkdir -p "$APP_DIR/static/outputs" "$APP_DIR/work"

echo "==> Installing the systemd service"
sudo cp "$APP_DIR/deploy/thumbsmith.service" /etc/systemd/system/thumbsmith.service
sudo systemctl daemon-reload
sudo systemctl enable --quiet thumbsmith

echo "==> Installing the nginx site"
sudo cp "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/thumbsmith
sudo ln -sf /etc/nginx/sites-available/thumbsmith /etc/nginx/sites-enabled/thumbsmith
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo "==> Installing the scratch-space cleaner"
# work/ holds downloaded videos and sampled frames — tens of MB per run, never read
# again once the thumbnail exists. On a 20 GB Lightsail disk this fills up in days.
sudo tee /etc/systemd/system/thumbsmith-cleanup.service >/dev/null <<'UNIT'
[Unit]
Description=Prune Thumbsmith scratch files older than a day

[Service]
Type=oneshot
ExecStart=/usr/bin/find /home/admin/thumbsmith/work -mindepth 1 -maxdepth 1 -type d -mtime +1 -exec rm -rf {} +
ExecStart=/usr/bin/find /home/admin/thumbsmith/static/outputs -type f -mtime +7 -delete
UNIT

sudo tee /etc/systemd/system/thumbsmith-cleanup.timer >/dev/null <<'UNIT'
[Unit]
Description=Run the Thumbsmith scratch cleaner daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now --quiet thumbsmith-cleanup.timer

if [ -f "$APP_DIR/.env" ]; then
    echo "==> .env present, starting the service"
    sudo systemctl restart thumbsmith
    sleep 3
    systemctl is-active --quiet thumbsmith \
        && echo "    thumbsmith is running" \
        || { echo "    FAILED — journalctl -u thumbsmith -n 40"; exit 1; }
else
    echo "==> No .env yet — copy it across, then: sudo systemctl restart thumbsmith"
fi

echo
echo "Done. ffmpeg: $(ffmpeg -version | head -1 | cut -d' ' -f1-3)"
