#!/usr/bin/env bash
# Install a YouTube cookie jar on the server so the link tab works there.
#
# Run from your own machine, not the server:
#
#   bash deploy/install-cookies.sh cookies.txt admin@13.212.81.206 ~/.ssh/lightsail_singapore.pem
#
# To produce cookies.txt: install the "Get cookies.txt LOCALLY" extension in Chrome or
# Firefox, sign in to YouTube, open youtube.com, and export in Netscape format.
#
# Use a throwaway Google account. These cookies are a live logged-in session, and Google
# may lock an account whose session suddenly appears from a server in another country.

set -euo pipefail

COOKIES="${1:?usage: install-cookies.sh <cookies.txt> <user@host> [ssh-key]}"
TARGET="${2:?usage: install-cookies.sh <cookies.txt> <user@host> [ssh-key]}"
KEY="${3:-}"

[ -f "$COOKIES" ] || { echo "No such file: $COOKIES"; exit 1; }

# A Netscape cookie jar starts with this header. Exporting in the wrong format is the
# usual mistake, and yt-dlp's failure for it looks identical to having no cookies at all.
if ! head -1 "$COOKIES" | grep -qi "netscape\|# HTTP Cookie File"; then
    echo "Warning: $COOKIES does not look like a Netscape cookie jar."
    echo "         Re-export choosing the Netscape / cookies.txt format, not JSON."
fi

if ! grep -q "youtube.com" "$COOKIES"; then
    echo "Error: no youtube.com cookies in that file. Export it with youtube.com open."
    exit 1
fi

SSH_ARGS=()
[ -n "$KEY" ] && SSH_ARGS=(-i "$KEY")

REMOTE_DIR="/home/admin/thumbsmith"

echo "==> Copying the cookie jar"
scp "${SSH_ARGS[@]}" -q "$COOKIES" "$TARGET:$REMOTE_DIR/cookies.txt"

echo "==> Pointing the app at it and restarting"
ssh "${SSH_ARGS[@]}" "$TARGET" bash -s <<'REMOTE'
set -euo pipefail
cd /home/admin/thumbsmith
chmod 600 cookies.txt

# Replace any existing line rather than appending a second one.
touch .env
grep -v '^YTDLP_COOKIES_FILE=' .env > .env.tmp || true
echo "YTDLP_COOKIES_FILE=/home/admin/thumbsmith/cookies.txt" >> .env.tmp
mv .env.tmp .env
chmod 600 .env

sudo systemctl restart thumbsmith
sleep 4
systemctl is-active --quiet thumbsmith && echo "    service running" || echo "    service FAILED"
REMOTE

echo
echo "Done. Try a link in the web UI."
echo "Cookies expire after days or weeks; re-run this when the link tab starts failing again."
