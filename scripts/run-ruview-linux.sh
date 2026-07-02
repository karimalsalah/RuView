#!/usr/bin/env bash
# RuView — live WiFi RSSI sensing on this Linux laptop.
#
# Uses the connected-AP RSSI (via `iw dev <iface> link/station dump`) as a
# REAL, passive sensing source — no ESP32, no root, never leaves the operating
# channel, connection stays up. A moving body modulates the RSSI; the server's
# temporal-variance pipeline turns that into live presence/motion in the
# observatory. NOTHING simulated.
#
# Usage:  ./scripts/run-ruview-linux.sh [iface] [tick_ms]
#   iface   wireless interface (default: auto-detect, falls back to wlo1)
#   tick_ms sensing tick in ms (default 100 = 10 Hz)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IFACE="${1:-wlo1}"
TICK="${2:-100}"
BIN="$REPO/v2/target/release/sensing-server"

if [ ! -x "$BIN" ]; then
  echo "Building sensing-server (optimized, native CPU) — first run only ..."
  ( cd "$REPO/v2" && RUSTFLAGS="-C target-cpu=native" \
      cargo build --release -p wifi-densepose-sensing-server )
fi

echo "Starting RuView sensing-server  (source=wifi  iface=$IFACE  tick=${TICK}ms)"
"$BIN" --source wifi --wifi-iface "$IFACE" --tick-ms "$TICK" &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' INT TERM
sleep 2

URL="http://localhost:8080/ui/observatory.html"
echo
echo "  Observatory : $URL"
echo "  REST        : http://localhost:8080/api/v1/sensing/latest"
echo "  WebSocket   : ws://localhost:8080/ws/sensing"
echo "  Stop        : kill $SRV   (or Ctrl-C)"
echo
xdg-open "$URL" >/dev/null 2>&1 || true
wait "$SRV"
