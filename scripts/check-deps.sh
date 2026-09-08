#!/usr/bin/env bash
# check-deps.sh — Verify (and optionally install) Nerdit system dependencies.
# Targets Ubuntu 22.04 / 24.04. Run standalone — no Python required.
#
# Usage:
#   bash scripts/check-deps.sh            # check only
#   bash scripts/check-deps.sh --install  # check + offer to install missing deps

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours & helpers
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

INSTALL_MODE=false
if [[ "${1:-}" == "--install" ]]; then
    INSTALL_MODE=true
fi

ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$1"; }
warn() { printf "  ${YELLOW}✗${RESET} %s\n" "$1"; }
fail() { printf "  ${RED}✗${RESET} %s\n" "$1"; }
info() { printf "  ${DIM}%s${RESET}\n" "$1"; }

# Ask the user to confirm an install.  Returns 0 if yes.
ask_install() {
    local name="$1"
    if [[ "$INSTALL_MODE" != true ]]; then
        return 1
    fi
    printf "\n${BOLD}Install %s?${RESET} [y/N] " "$name"
    read -r answer
    [[ "$answer" =~ ^[Yy]$ ]]
}

# Track overall status
ALL_OK=true
mark_missing() { ALL_OK=false; }

# ---------------------------------------------------------------------------
# 1. Python 3.11+
# ---------------------------------------------------------------------------
printf "\n${BOLD}Checking dependencies...${RESET}\n\n"

check_python() {
    if command -v python3 &>/dev/null; then
        local ver
        ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        local major minor
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [[ "$major" -ge 3 && "$minor" -ge 11 ]]; then
            ok "Python $ver"
            return 0
        else
            warn "Python $ver found (need ≥3.11)"
        fi
    else
        warn "Python 3 not found"
    fi
    return 1
}

if ! check_python; then
    mark_missing
    if ask_install "Python 3.11"; then
        sudo apt-get update && sudo apt-get install -y python3.11
        check_python || true
    else
        info "Install: sudo apt install python3.11"
        info "For older Ubuntu: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Optional vendor-neutral GPU discovery
# ---------------------------------------------------------------------------
check_zml_smi() {
    if ! command -v zml-smi &>/dev/null; then
        info "zml-smi not found (optional; Nerdit will fall back to pynvml)"
        return 1
    fi
    if zml-smi --json 2>/dev/null \
        | python3 -c 'import json,sys; assert isinstance(json.load(sys.stdin).get("devices"), list)'; then
        ok "zml-smi JSON backend"
        return 0
    fi
    info "zml-smi found but --json failed (optional; Nerdit will fall back to pynvml)"
    return 1
}

check_zml_smi || true

# ---------------------------------------------------------------------------
# 3. NVIDIA Driver / nvidia-smi
# ---------------------------------------------------------------------------
check_nvidia_driver() {
    if command -v nvidia-smi &>/dev/null; then
        local driver_ver
        driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [[ -n "$driver_ver" ]]; then
            local major
            major=$(echo "$driver_ver" | cut -d. -f1)
            if [[ "$major" -ge 535 ]]; then
                ok "NVIDIA Driver $driver_ver"
                return 0
            else
                warn "NVIDIA Driver $driver_ver (need ≥535)"
            fi
        else
            warn "nvidia-smi present but could not query driver version"
        fi
    else
        warn "nvidia-smi not found"
    fi
    return 1
}

if ! check_nvidia_driver; then
    mark_missing
    if ask_install "NVIDIA Driver (≥535)"; then
        sudo apt-get update && sudo apt-get install -y nvidia-driver-535
        info "A reboot may be required after driver installation."
        check_nvidia_driver || true
    else
        info "Install: sudo apt install nvidia-driver-535  (reboot after)"
    fi
fi

# ---------------------------------------------------------------------------
# 4. libnvidia-ml.so.1
# ---------------------------------------------------------------------------
check_nvidia_lib() {
    local dirs=("/usr/lib/x86_64-linux-gnu" "/usr/lib64" "/usr/lib")
    for d in "${dirs[@]}"; do
        if [[ -f "$d/libnvidia-ml.so.1" ]]; then
            ok "libnvidia-ml.so.1 ($d)"
            return 0
        fi
    done
    warn "libnvidia-ml.so.1 not found"
    return 1
}

if ! check_nvidia_lib; then
    mark_missing
    info "This library is provided by the NVIDIA driver package."
    info "If the driver is installed, try: sudo ldconfig"
fi

# ---------------------------------------------------------------------------
# 5. Docker Engine
# ---------------------------------------------------------------------------
check_docker() {
    if command -v docker &>/dev/null && docker info &>/dev/null; then
        local docker_ver
        docker_ver=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
        ok "Docker Engine $docker_ver"
        return 0
    else
        if command -v docker &>/dev/null; then
            warn "Docker installed but not running or no permission"
        else
            warn "Docker not found"
        fi
    fi
    return 1
}

if ! check_docker; then
    mark_missing
    if ask_install "Docker Engine"; then
        curl -fsSL https://get.docker.com | sudo sh
        sudo usermod -aG docker "$USER"
        info "You may need to log out and back in for group changes to take effect."
        check_docker || true
    else
        info "Install: curl -fsSL https://get.docker.com | sudo sh"
        info "Then:    sudo usermod -aG docker \$USER  (log out/in after)"
    fi
fi

# ---------------------------------------------------------------------------
# 6. NVIDIA Container Toolkit
# ---------------------------------------------------------------------------
check_nvidia_toolkit() {
    if ! command -v docker &>/dev/null || ! docker info &>/dev/null; then
        warn "NVIDIA Container Toolkit — skipped (Docker not available)"
        return 1
    fi
    local runtimes
    runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null || echo "")
    if echo "$runtimes" | grep -q '"nvidia"'; then
        ok "NVIDIA Container Toolkit"
        return 0
    else
        warn "NVIDIA Container Toolkit not found in Docker runtimes"
    fi
    return 1
}

if ! check_nvidia_toolkit; then
    mark_missing
    if ask_install "NVIDIA Container Toolkit"; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        check_nvidia_toolkit || true
    else
        info "Install: sudo apt install nvidia-container-toolkit"
        info "Then:    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    fi
fi

# ---------------------------------------------------------------------------
# 7. nerdit-runtime:0.1 image
# ---------------------------------------------------------------------------
check_runtime_image() {
    if ! command -v docker &>/dev/null || ! docker info &>/dev/null; then
        warn "nerdit-runtime:0.1 — skipped (Docker not available)"
        return 1
    fi
    if docker image inspect nerdit-runtime:0.1 &>/dev/null; then
        ok "nerdit-runtime:0.1 image"
        return 0
    else
        warn "nerdit-runtime:0.1 image not found"
    fi
    return 1
}

if ! check_runtime_image; then
    mark_missing
    # Try to find the Dockerfile relative to this script
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    DOCKERFILE="${SCRIPT_DIR}/../docker/Dockerfile"
    if [[ -f "$DOCKERFILE" ]]; then
        if ask_install "nerdit-runtime:0.1 image (docker build)"; then
            docker build -t nerdit-runtime:0.1 "$(dirname "$DOCKERFILE")"
            check_runtime_image || true
        else
            info "Build: docker build -t nerdit-runtime:0.1 docker/"
        fi
    else
        info "Build: docker build -t nerdit-runtime:0.1 docker/"
        info "(run from the repository root)"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
printf "\n${BOLD}─────────────────────────────${RESET}\n"
if [[ "$ALL_OK" == true ]]; then
    printf "${GREEN}${BOLD}All dependencies are installed.${RESET}\n"
    exit 0
else
    printf "${YELLOW}${BOLD}Some dependencies are missing — see above.${RESET}\n"
    if [[ "$INSTALL_MODE" != true ]]; then
        printf "${DIM}Run with --install to interactively install missing dependencies.${RESET}\n"
    fi
    exit 1
fi
