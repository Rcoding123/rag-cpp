#!/usr/bin/env bash
# Bootstrap the software-factory box (Ubuntu 22.04+/Debian 12+).
#
# Run once as root:
#     sudo bash factory_bootstrap.sh [username]   # default username: factory
#
# This box is fully separate: it has no access to, and no keys for, any other server.
#
# What it does:
#   1. Updates the OS; installs C/C++, Python, Java, monitoring and security basics.
#   2. Creates a user with passwordless sudo (this box is for building anything).
#   3. Hardens SSH (key-only, no root login) ONLY if that user has a key - avoids lockout.
#   4. Firewall: allow SSH only. UTC clock + chrony. Automatic security updates.
#   5. Docker Engine + Compose (official repo). Published ports bind to 127.0.0.1 by
#      default, because Docker otherwise bypasses the firewall.
#   6. Node.js LTS (NodeSource repo), Go (official tarball), Rust (rustup), uv (pipx).
#   7. /srv/factory/projects workspace.
#   8. Weekly cleanup of old Docker images/build cache and system journal.
#
# It installs language toolchains only - no AI tools or agent CLIs of any kind.
set -euo pipefail

USERNAME="${1:-factory}"
NODE_MAJOR="${NODE_MAJOR:-24}"    # Node.js LTS line

log() { printf '\n==> %s\n' "$*"; }

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo bash $0)"; exit 1; }
command -v apt-get >/dev/null || { echo "This script supports Debian/Ubuntu (apt) only."; exit 1; }
. /etc/os-release
echo "Detected: ${PRETTY_NAME}"
case "$(dpkg --print-architecture)" in
  amd64) GOARCH=amd64 ;;
  arm64) GOARCH=arm64 ;;
  *) echo "Unsupported architecture"; exit 1 ;;
esac

# ---------------------------------------------------------------- 1. packages
log "Updating OS and installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
# --with-new-pkgs: also install new dependencies, so a library is never upgraded
# while its daemon is kept back (that broke fwupd). It never removes packages.
apt-get upgrade -y --with-new-pkgs
apt-get install -y \
  build-essential cmake ninja-build clang lld gdb pkg-config ccache \
  git git-lfs curl wget unzip zip zstd jq rsync tmux htop btop iotop sysstat \
  ca-certificates gnupg \
  python3 python3-venv python3-dev python3-pip pipx \
  default-jdk maven \
  ufw fail2ban chrony unattended-upgrades

# ---------------------------------------------------------------- 2. user
log "Creating user '${USERNAME}' (passwordless sudo)"
if ! id "${USERNAME}" &>/dev/null; then
  adduser --disabled-password --gecos "" "${USERNAME}"
fi
usermod -aG sudo "${USERNAME}"
echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-${USERNAME}"
chmod 440 "/etc/sudoers.d/90-${USERNAME}"
visudo -cf "/etc/sudoers.d/90-${USERNAME}" >/dev/null || { rm -f "/etc/sudoers.d/90-${USERNAME}"; echo "sudoers check failed"; exit 1; }

USER_HOME=$(getent passwd "${USERNAME}" | cut -d: -f6)
install -d -m 700 -o "${USERNAME}" -g "${USERNAME}" "${USER_HOME}/.ssh"
if [[ -s /root/.ssh/authorized_keys ]]; then
  cat /root/.ssh/authorized_keys >> "${USER_HOME}/.ssh/authorized_keys"
  sort -u -o "${USER_HOME}/.ssh/authorized_keys" "${USER_HOME}/.ssh/authorized_keys"
fi
touch "${USER_HOME}/.ssh/authorized_keys"
chown "${USERNAME}:${USERNAME}" "${USER_HOME}/.ssh/authorized_keys"
chmod 600 "${USER_HOME}/.ssh/authorized_keys"

# ---------------------------------------------------------------- 3. SSH hardening
if [[ -s "${USER_HOME}/.ssh/authorized_keys" ]]; then
  log "Hardening SSH (key-only login, no root login)"
  # sshd keeps the FIRST value it reads and loads drop-ins in name order, so this file
  # must sort first (00-) or e.g. 50-cloud-init.conf's "PasswordAuthentication yes" wins.
  rm -f /etc/ssh/sshd_config.d/99-hardening.conf
  cat > /etc/ssh/sshd_config.d/00-hardening.conf <<'EOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
PubkeyAuthentication yes
EOF
  if sshd -t; then
    systemctl reload ssh 2>/dev/null || systemctl reload sshd
    eff=$(sshd -T 2>/dev/null | awk '/^(passwordauthentication|permitrootlogin) /{printf "%s=%s ", $1, $2}')
    [[ $eff == *"passwordauthentication=no"* && $eff == *"permitrootlogin=no"* ]] \
      && echo "SSH effective settings: ${eff}" \
      || echo "WARNING: SSH hardening not fully effective: ${eff}"
  else
    rm -f /etc/ssh/sshd_config.d/00-hardening.conf
    echo "sshd config test failed - hardening reverted."
  fi
else
  echo "WARNING: ${USERNAME} has no SSH key, so SSH hardening was SKIPPED to avoid lockout."
  echo "Add a key to ${USER_HOME}/.ssh/authorized_keys and re-run this script."
fi

# ---------------------------------------------------------------- 4. firewall, time, updates
log "Firewall (allow SSH only), UTC clock, chrony, unattended security upgrades"
ufw allow OpenSSH
ufw --force enable
timedatectl set-timezone UTC
systemctl enable --now chrony
systemctl enable --now fail2ban
dpkg-reconfigure -f noninteractive unattended-upgrades

# ---------------------------------------------------------------- 5. Docker
log "Installing Docker Engine + Compose"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL "https://download.docker.com/linux/${ID}/gpg" -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
install -d /etc/docker
cat > /etc/docker/daemon.json <<'EOF'
{
  "ip": "127.0.0.1",
  "log-driver": "json-file",
  "log-opts": { "max-size": "50m", "max-file": "3" }
}
EOF
systemctl enable docker
systemctl restart docker
usermod -aG docker "${USERNAME}"

# ---------------------------------------------------------------- 6. language toolchains
log "Installing Node.js ${NODE_MAJOR}.x"
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
  | gpg --dearmor --yes -o /etc/apt/keyrings/nodesource.gpg
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
  > /etc/apt/sources.list.d/nodesource.list
apt-get update -y
apt-get install -y nodejs
corepack enable 2>/dev/null || echo "corepack not available in this Node version - skipped"

log "Installing Go (latest stable)"
GO_VERSION=$(curl -fsSL "https://go.dev/VERSION?m=text" | head -n1)
curl -fsSL "https://go.dev/dl/${GO_VERSION}.linux-${GOARCH}.tar.gz" -o /tmp/go.tgz
rm -rf /usr/local/go
tar -C /usr/local -xzf /tmp/go.tgz
rm -f /tmp/go.tgz
echo 'export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin' > /etc/profile.d/go.sh

log "Installing Rust (rustup) and uv for ${USERNAME}"
sudo -u "${USERNAME}" -H bash -c '
  set -e
  if ! command -v rustup >/dev/null && [[ ! -x "$HOME/.cargo/bin/rustup" ]]; then
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default
  fi
  pipx install uv >/dev/null 2>&1 || pipx upgrade uv >/dev/null 2>&1 || true
  pipx ensurepath >/dev/null 2>&1 || true
'

# ---------------------------------------------------------------- 7. workspace
log "Creating /srv/factory workspace"
install -d -o "${USERNAME}" -g "${USERNAME}" /srv/factory /srv/factory/projects
ln -sfn /srv/factory/projects "${USER_HOME}/projects"
chown -h "${USERNAME}:${USERNAME}" "${USER_HOME}/projects"

# ---------------------------------------------------------------- 8. weekly cleanup
log "Scheduling weekly cleanup (Docker images/cache older than 7 days, journal > 1 GB)"
cat > /etc/systemd/system/factory-cleanup.service <<'EOF'
[Unit]
Description=Weekly disk cleanup for the factory box
[Service]
Type=oneshot
ExecStart=/usr/bin/docker system prune -af --filter until=168h
ExecStart=/usr/bin/docker builder prune -af --filter until=168h
ExecStart=/usr/bin/journalctl --vacuum-size=1G
EOF
cat > /etc/systemd/system/factory-cleanup.timer <<'EOF'
[Unit]
Description=Run factory cleanup weekly
[Timer]
OnCalendar=Sun 04:00
Persistent=true
[Install]
WantedBy=timers.target
EOF
systemctl daemon-reload
systemctl enable --now factory-cleanup.timer

# ---------------------------------------------------------------- summary
log "Done"
MEM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
cat <<EOF
Host:      $(hostname)  (${PRETTY_NAME})
CPUs:      $(nproc)   RAM: ${MEM_GB} GB
Disk:      $(df -h / | awk 'NR==2 {print $2" total, "$4" free"}')
User:      ${USERNAME} (passwordless sudo, docker group)
Docker:    $(docker --version 2>/dev/null)
Compose:   $(docker compose version 2>/dev/null)
Node:      $(node --version 2>/dev/null)
Go:        ${GO_VERSION}
Java:      $(java -version 2>&1 | head -n1)
Python:    $(python3 --version)
Workspace: /srv/factory/projects  (~/projects)
Cleanup:   weekly, Sunday 04:00 UTC

Next:
  1. From a NEW terminal, confirm login works:  ssh ${USERNAME}@<this-box-ip>
     (keep this root session open until that works)
  Note: Docker ports are published on 127.0.0.1 only. Reach them with an SSH tunnel,
        or publish explicitly on 0.0.0.0 AND open the port in ufw if you really need it public.
EOF
