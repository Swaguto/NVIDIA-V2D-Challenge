#!/bin/bash
# VNC stack for the shared box (netmode=host). Port 5900 opens on the VM host.
set -x
export DISPLAY=:99

Xvfb :99 -screen 0 1280x720x24 -nolisten tcp &
XPID=$!
sleep 3

x11vnc -rfbport 5900 -display :99 -auth guess -scale 0.75 \
  -tight -zlib -ncache 10 -nopw -forever &
VPID=$!
sleep 2

openbox &
OPID=$!

echo "VNC_STACK_UP xvfb=$XPID x11vnc=$VPID openbox=$OPID"
# Keep the container alive as the foreground process
tail -f /dev/null