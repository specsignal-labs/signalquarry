#!/bin/sh
# Prepare a dedicated VM for one paper deployment (run as root on the VM; review first).
#   sh host-bootstrap.sh ALIAS
# One unix user per deployment, no sudo, no docker group; state under /srv/signalquarry/ALIAS.
set -eu
alias="$1"; user="sq-$alias"
id "$user" >/dev/null 2>&1 || useradd --create-home --shell /usr/sbin/nologin "$user"
install -d -o "$user" -g "$user" -m 0750 "/srv/signalquarry/$alias" "/srv/signalquarry/$alias/releases"
echo "Next, as $user: store keys with systemd-creds (LoadCredential), install uv, then"
echo "  sqy paper schedule --alias $alias --target systemd   # review and install the units"
