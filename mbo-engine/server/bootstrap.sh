#!/usr/bin/env bash
# Bootstrap the research/compute box (Ubuntu 22.04+/Debian 12+).
#
# Run once as root:
#     sudo bash bootstrap.sh [username]          # default username: quant
#
# What it does:
#   1. Updates the OS; installs C++ toolchain, Python, monitoring and security basics.
#   2. Creates a sudo user (copies root's SSH keys to it).
#   3. Hardens SSH (key-only, no root login) ONLY if that user has a key - avoids lockout.
#   4. Firewall: allow SSH only.  Time sync (chrony) + UTC clock - matters for MBO timestamps.
#   5. Creates /srv/quant data layout and a Python venv with databento + data stack.
#   6. Creates two systemd slices so live jobs keep reserved cores:
#        quant-live.slice   -> pinned to the first LIVE_CORES cores, high CPU weight
#        quant-batch.slice  -> remaining cores, low CPU weight, memory capped
#      How to run jobs in them is printed under "Usage" at the end.
set -euo pipefail

USERNAME="${1:-quant}"
LIVE_CORES="${LIVE_CORES:-4}"          # cores reserved for live engine
BATCH_MEM_PCT="${BATCH_MEM_PCT:-80}"   # batch jobs may use at most this % of RAM
ROOT_DIR="/srv/quant"

log() { printf '\n==> %s\n' "$*"; }

[[ $EUID -eq 0 ]] || { echo "Run as root (sudo bash $0)"; exit 1; }
command -v apt-get >/dev/null || { echo "This script supports Debian/Ubuntu (apt) only."; exit 1; }
. /etc/os-release
echo "Detected: ${PRETTY_NAME}"

# ---------------------------------------------------------------- 1. packages
log "Updating OS and installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
# --with-new-pkgs: also install new dependencies, so a library is never upgraded
# while its daemon is kept back (that broke fwupd). It never removes packages.
apt-get upgrade -y --with-new-pkgs
apt-get install -y \
  build-essential cmake ninja-build clang lld gdb pkg-config ccache \
  git curl wget unzip zstd jq rsync tmux htop btop iotop sysstat smartmontools nvme-cli \
  python3 python3-venv python3-dev python3-pip \
  ufw fail2ban chrony unattended-upgrades

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || { echo "Python ${PYVER} found; databento needs 3.11+. Upgrade the OS or install a newer Python."; exit 1; }

# ---------------------------------------------------------------- 2. user
log "Creating user '${USERNAME}'"
if ! id "${USERNAME}" &>/dev/null; then
  adduser --disabled-password --gecos "" "${USERNAME}"
fi
usermod -aG sudo "${USERNAME}"
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
  echo "Passwordless sudo is NOT set; set a password for ${USERNAME} with: passwd ${USERNAME}"
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

# ---------------------------------------------------------------- 5. data layout + venv
log "Creating ${ROOT_DIR} layout and Python venv"
install -d -o "${USERNAME}" -g "${USERNAME}" \
  "${ROOT_DIR}" "${ROOT_DIR}/data/raw" "${ROOT_DIR}/data/features" \
  "${ROOT_DIR}/results" "${ROOT_DIR}/models" "${ROOT_DIR}/logs" "${ROOT_DIR}/code"

if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  printf '# Databento API key. Never commit or share this file.\nDATABENTO_API_KEY=paste_your_key_here\n' > "${ROOT_DIR}/.env"
fi
chown "${USERNAME}:${USERNAME}" "${ROOT_DIR}/.env"
chmod 600 "${ROOT_DIR}/.env"

sudo -u "${USERNAME}" bash -c "
  set -e
  [[ -d '${ROOT_DIR}/.venv' ]] || python3 -m venv '${ROOT_DIR}/.venv'
  '${ROOT_DIR}/.venv/bin/pip' install -q --upgrade pip wheel
  '${ROOT_DIR}/.venv/bin/pip' install -q databento numpy pandas polars pyarrow duckdb numba pybind11 nanobind
  '${ROOT_DIR}/.venv/bin/python' -c 'import databento; print(\"databento\", databento.__version__)'
"

# ---------------------------------------------------------------- 6. systemd slices
log "Creating CPU/memory slices for live vs batch work"
NCPU=$(nproc)

# L3 cache size (KB) seen by a logical CPU; 0 if unknown.
l3_kb() {
  local f="/sys/devices/system/cpu/cpu$1/cache/index3/size" v
  [[ -r $f ]] || { echo 0; return; }
  v=$(<"$f")
  case $v in *K) echo "${v%K}" ;; *M) echo $(( ${v%M} * 1024 )) ;; *) echo "${v//[!0-9]/}" ;; esac
}
cpu_rows() { lscpu -p=CPU,CORE | grep -v '^#'; }   # "logical_cpu,physical_core"

# >>> core-pick
# Reserve LIVE_CORES *physical* cores (with all their SMT threads) for live work, so
# batch jobs never share a physical core with the live engine. On CPUs with uneven L3
# (X3D V-cache parts) the live cores are taken from the CCD with the largest L3.
declare -A CORE_CPUS=() CORE_L3=()
while IFS=, read -r cpu core; do
  CORE_CPUS[$core]+="${CORE_CPUS[$core]:+,}${cpu}"
  CORE_L3[$core]=$(l3_kb "$cpu")
done < <(cpu_rows)
NCORES=${#CORE_CPUS[@]}
MAX_L3=$(printf '%s\n' "${CORE_L3[@]}" | sort -n | tail -n1)
LIVE_SET=""; BATCH_SET=""
if (( NCORES > LIVE_CORES + 1 )); then
  LIVE_IDS=" $(for c in "${!CORE_L3[@]}"; do [[ ${CORE_L3[$c]} == "$MAX_L3" ]] && echo "$c"; done \
               | sort -n | head -n "$LIVE_CORES" | tr '\n' ' ')"
  for c in $(printf '%s\n' "${!CORE_CPUS[@]}" | sort -n); do
    if [[ $LIVE_IDS == *" $c "* ]]; then
      LIVE_SET+="${LIVE_SET:+,}${CORE_CPUS[$c]}"
    else
      BATCH_SET+="${BATCH_SET:+,}${CORE_CPUS[$c]}"
    fi
  done
else
  LIVE_SET="0-$((NCPU - 1))"; BATCH_SET="0-$((NCPU - 1))"
  echo "Only ${NCORES} physical cores - no cores reserved, priorities only."
fi
# <<< core-pick
MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
BATCH_MEM_MB=$(( MEM_KB * BATCH_MEM_PCT / 100 / 1024 ))

cat > /etc/systemd/system/quant-live.slice <<EOF
[Unit]
Description=Live trading/engine workloads (reserved cores)
[Slice]
AllowedCPUs=${LIVE_SET}
CPUWeight=1000
IOWeight=1000
EOF

cat > /etc/systemd/system/quant-batch.slice <<EOF
[Unit]
Description=Batch workloads: backtests, RL training, feature builds
[Slice]
AllowedCPUs=${BATCH_SET}
CPUWeight=50
IOWeight=50
MemoryMax=${BATCH_MEM_MB}M
EOF
systemctl daemon-reload

# ---------------------------------------------------------------- summary
log "Done"
cat <<EOF
Host:        $(hostname)  (${PRETTY_NAME})
CPU:         $(lscpu | awk -F: '/Model name/ {gsub(/^ +/,"",$2); print $2; exit}')
Cores:       ${NCORES} physical / ${NCPU} threads   largest L3 per core group: $(( MAX_L3 / 1024 )) MB
Live CPUs:   ${LIVE_SET}   (${LIVE_CORES} physical cores incl. SMT threads)
Batch CPUs:  ${BATCH_SET}
RAM:         $(( MEM_KB / 1024 / 1024 )) GB   batch memory cap: ${BATCH_MEM_MB} MB
Disk:        $(df -h "${ROOT_DIR}" | awk 'NR==2 {print $2" total, "$4" free"}')
Python:      ${PYVER}   venv: ${ROOT_DIR}/.venv
User:        ${USERNAME}
Data root:   ${ROOT_DIR}  (data/raw, data/features, results, models, logs, code)

Usage:
  Batch job (backtest, training):
    sudo systemd-run --slice=quant-batch.slice --uid=${USERNAME} -p WorkingDirectory=${ROOT_DIR} \\
         --unit=bt-\$(date +%s) ${ROOT_DIR}/.venv/bin/python code/your_job.py
  Live service: create a unit with  Slice=quant-live.slice  under [Service].
  Check usage:  systemd-cgtop

Next:
  1. In THIS root session, set ${USERNAME}'s sudo password:  passwd ${USERNAME}
  2. From a NEW terminal, confirm you can log in:  ssh ${USERNAME}@<server-ip>
     (keep this root session open until that works)
  3. Put your Databento key in ${ROOT_DIR}/.env
EOF
