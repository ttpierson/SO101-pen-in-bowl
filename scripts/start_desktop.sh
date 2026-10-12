#!/bin/bash
# Start a virtual X desktop (Xvfb :1, 1920x1080) and serve it via noVNC on localhost:6080.
# Isaac Sim still renders on the GPU (Vulkan) and presents into this display.
export DISPLAY=:1
pgrep -x Xvfb >/dev/null || { nohup Xvfb :1 -screen 0 1920x1080x24 +extension GLX +extension RANDR -nolisten tcp > /tmp/xvfb.log 2>&1 & sleep 2; }
pgrep -x xfce4-session >/dev/null || { nohup dbus-launch startxfce4 > /tmp/xfce.log 2>&1 & sleep 4; }
xfconf-query -c xfwm4 -p /general/use_compositing -s false 2>/dev/null
xset s off -dpms 2>/dev/null
pgrep -x x11vnc >/dev/null || nohup x11vnc -display :1 -forever -shared -nopw -localhost -rfbport 5900 > /tmp/x11vnc.log 2>&1 &
pgrep -f "[w]ebsockify.*6080" >/dev/null || nohup websockify --web /usr/share/novnc 6080 localhost:5900 > /tmp/novnc.log 2>&1 &
sleep 2
echo "desktop up on :1, noVNC on localhost:6080"
