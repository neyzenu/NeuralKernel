#!/bin/bash

# kCompanion setup script
# This script handles installation, configuration and management
# of both the kernel module and the AI daemon

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
KCOMP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${KCOMP_DIR}/ai_models"
LOG_DIR="${KCOMP_DIR}/ai_logs"
SERVICE_DIR="/etc/systemd/system"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# Print header
echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}      kCompanion AI System Setup Script       ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Function to check requirements
check_requirements() {
    echo -e "${YELLOW}Checking system requirements...${NC}"
    
    # Check kernel version
    KERNEL_VERSION=$(uname -r)
    MAJOR_VERSION=$(echo $KERNEL_VERSION | cut -d. -f1)
    MINOR_VERSION=$(echo $KERNEL_VERSION | cut -d. -f2)
    
    if [ "$MAJOR_VERSION" -lt 6 ] || [ "$MAJOR_VERSION" -eq 6 -a "$MINOR_VERSION" -lt 8 ]; then
        echo -e "${RED}Warning: Kernel version $KERNEL_VERSION detected.${NC}"
        echo -e "${RED}Recommended: Linux kernel 6.8+ for optimal functionality.${NC}"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}Kernel version $KERNEL_VERSION: OK${NC}"
    fi
    
    # Check for build tools
    if ! command -v make &> /dev/null; then
        echo -e "${RED}Error: 'make' command not found.${NC}"
        echo -e "${YELLOW}Installing build-essential package...${NC}"
        apt-get update && apt-get install -y build-essential
    else
        echo -e "${GREEN}Build tools: OK${NC}"
    fi
    
    # Check for kernel headers
    if [ ! -d "/lib/modules/$(uname -r)/build" ]; then
        echo -e "${RED}Error: Kernel headers not found.${NC}"
        echo -e "${YELLOW}Installing linux-headers package...${NC}"
        apt-get update && apt-get install -y linux-headers-$(uname -r)
    else
        echo -e "${GREEN}Kernel headers: OK${NC}"
    fi
    
    # Check for Python 3.8+
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 not found.${NC}"
        echo -e "${YELLOW}Installing Python 3...${NC}"
        apt-get update && apt-get install -y python3 python3-pip
    else
        PY_VERSION=$(python3 --version | cut -d' ' -f2)
        PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
        PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)
        
        if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MAJOR" -eq 3 -a "$PY_MINOR" -lt 8 ]; then
            echo -e "${RED}Python version $PY_VERSION detected. Version 3.8+ required.${NC}"
            exit 1
        else
            echo -e "${GREEN}Python $PY_VERSION: OK${NC}"
        fi
    fi
    
    # Check for NVIDIA GPU and CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: NVIDIA GPU drivers not detected.${NC}"
        echo -e "${YELLOW}AI component will run in CPU mode (slower).${NC}"
    else
        echo -e "${GREEN}NVIDIA GPU detected.${NC}"
        
        # Check CUDA
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            echo -e "${GREEN}CUDA available for PyTorch: OK${NC}"
        else
            echo -e "${YELLOW}Warning: PyTorch with CUDA support not detected.${NC}"
            echo -e "${YELLOW}AI component will run in CPU mode (slower).${NC}"
            echo -e "${YELLOW}To enable GPU acceleration, install PyTorch with CUDA:${NC}"
            echo -e "${YELLOW}pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118${NC}"
        fi
    fi
}

# Function to install Python dependencies
install_python_deps() {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    
    # Create virtual environment if not exist
    if [ ! -d "${KCOMP_DIR}/venv" ]; then
        echo -e "${BLUE}Creating virtual environment...${NC}"
        apt-get install -y python3-venv
        python3 -m venv "${KCOMP_DIR}/venv"
    fi
    
    # Activate virtual environment and install dependencies
    echo -e "${BLUE}Installing dependencies...${NC}"
    source "${KCOMP_DIR}/venv/bin/activate"
    
    # Install PyTorch with CUDA if available
    if command -v nvidia-smi &> /dev/null; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch torchvision torchaudio
    fi
    
    # Install other dependencies
    pip install numpy pandas matplotlib
    
    # Deactivate virtual environment
    deactivate
    
    echo -e "${GREEN}Python dependencies installed.${NC}"
}

# Function to build kernel module
build_kernel_module() {
    echo -e "${YELLOW}Building kernel module...${NC}"
    cd "${KCOMP_DIR}"
    make clean
    make
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Kernel module built successfully.${NC}"
        return 0
    else
        echo -e "${RED}Failed to build kernel module.${NC}"
        return 1
    fi
}

# Function to install kernel module
install_kernel_module() {
    echo -e "${YELLOW}Installing kernel module...${NC}"
    
    # Check if already loaded
    if lsmod | grep -q "kcompanion"; then
        echo -e "${BLUE}Removing existing kcompanion module...${NC}"
        rmmod kcompanion
    fi
    
    # Install the module
    insmod "${KCOMP_DIR}/kcompanion.ko"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Kernel module installed successfully.${NC}"
        
        # Set permissions for device file
        if [ -e "/dev/kcompanion" ]; then
            chmod 666 /dev/kcompanion
            echo -e "${GREEN}Device permissions set.${NC}"
        else
            echo -e "${RED}Device file not created. Something went wrong.${NC}"
            return 1
        fi
        return 0
    else
        echo -e "${RED}Failed to install kernel module.${NC}"
        return 1
    fi
}

# Create systemd service file for AI daemon
create_service_file() {
    echo -e "${YELLOW}Creating systemd service file...${NC}"
    
    cat > "${SERVICE_DIR}/kcompanion-ai.service" << EOF
[Unit]
Description=kCompanion AI System
After=network.target

[Service]
ExecStart=${KCOMP_DIR}/venv/bin/python3 ${KCOMP_DIR}/kcompanion_ai.py
WorkingDirectory=${KCOMP_DIR}
Restart=always
RestartSec=5
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    echo -e "${GREEN}Service file created.${NC}"
}

# Function to start the system
start_system() {
    echo -e "${YELLOW}Starting kCompanion system...${NC}"
    
    # Load kernel module if not loaded
    if ! lsmod | grep -q "kcompanion"; then
        if ! install_kernel_module; then
            return 1
        fi
    fi
    
    # Create directories if not exist
    mkdir -p "${MODEL_DIR}"
    mkdir -p "${LOG_DIR}"
    
    # Start AI daemon as a service if service file exists
    if [ -f "${SERVICE_DIR}/kcompanion-ai.service" ]; then
        systemctl start kcompanion-ai
        systemctl status kcompanion-ai --no-pager
    else
        # Start AI daemon directly
        source "${KCOMP_DIR}/venv/bin/activate"
        nohup python3 "${KCOMP_DIR}/kcompanion_ai.py" > "${LOG_DIR}/ai_daemon.log" 2>&1 &
        echo $! > "${KCOMP_DIR}/ai_daemon.pid"
        deactivate
        echo -e "${GREEN}AI daemon started (PID: $(cat ${KCOMP_DIR}/ai_daemon.pid))${NC}"
    fi
    
    echo -e "${GREEN}kCompanion system started.${NC}"
}

# Function to stop the system
stop_system() {
    echo -e "${YELLOW}Stopping kCompanion system...${NC}"
    
    # Stop AI daemon
    if [ -f "${SERVICE_DIR}/kcompanion-ai.service" ]; then
        systemctl stop kcompanion-ai
    elif [ -f "${KCOMP_DIR}/ai_daemon.pid" ]; then
        PID=$(cat "${KCOMP_DIR}/ai_daemon.pid")
        if ps -p $PID > /dev/null; then
            kill $PID
            echo -e "${GREEN}AI daemon stopped.${NC}"
        else
            echo -e "${YELLOW}AI daemon not running.${NC}"
        fi
        rm -f "${KCOMP_DIR}/ai_daemon.pid"
    else
        echo -e "${YELLOW}AI daemon not running or PID file not found.${NC}"
    fi
    
    # Unload kernel module
    if lsmod | grep -q "kcompanion"; then
        rmmod kcompanion
        echo -e "${GREEN}Kernel module unloaded.${NC}"
    else
        echo -e "${YELLOW}Kernel module not loaded.${NC}"
    fi
    
    echo -e "${GREEN}kCompanion system stopped.${NC}"
}

# Function to build client application
build_client() {
    echo -e "${YELLOW}Building client application...${NC}"
    cd "${KCOMP_DIR}"
    make -f client_makefile
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Client application built successfully.${NC}"
        # Create a symlink in /usr/local/bin for easy access
        if [ ! -L "/usr/local/bin/kcompanion" ]; then
            ln -s "${KCOMP_DIR}/kcompanion_client" /usr/local/bin/kcompanion
            echo -e "${GREEN}Created symlink: /usr/local/bin/kcompanion${NC}"
        fi
        return 0
    else
        echo -e "${RED}Failed to build client application.${NC}"
        return 1
    fi
}

# Function to perform complete installation
install_system() {
    echo -e "${YELLOW}Installing kCompanion system...${NC}"
    
    # Check requirements
    check_requirements
    
    # Build kernel module
    if ! build_kernel_module; then
        return 1
    fi
    
    # Install Python dependencies
    install_python_deps
    
    # Create service file
    create_service_file
    
    # Enable service to start on boot
    systemctl enable kcompanion-ai
    
    # Create directories
    mkdir -p "${MODEL_DIR}"
    mkdir -p "${LOG_DIR}"
    
    # Build client application
    build_client
    
    echo -e "${GREEN}kCompanion system installed successfully.${NC}"
    echo -e "${BLUE}You can now start the system with: sudo $0 start${NC}"
}

# Function to uninstall the system
uninstall_system() {
    echo -e "${YELLOW}Uninstalling kCompanion system...${NC}"
    
    # Stop the system first
    stop_system
    
    # Remove service file
    if [ -f "${SERVICE_DIR}/kcompanion-ai.service" ]; then
        systemctl disable kcompanion-ai
        rm "${SERVICE_DIR}/kcompanion-ai.service"
        systemctl daemon-reload
        echo -e "${GREEN}Service file removed.${NC}"
    fi
    
    # Clean up compiled files
    cd "${KCOMP_DIR}"
    make clean
    
    # Remove client symlink
    if [ -L "/usr/local/bin/kcompanion" ]; then
        rm /usr/local/bin/kcompanion
        echo -e "${GREEN}Removed client symlink.${NC}"
    fi
    
    echo -e "${YELLOW}Do you want to remove AI models and logs? (y/n)${NC}"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${MODEL_DIR}" "${LOG_DIR}"
        echo -e "${GREEN}AI models and logs removed.${NC}"
    fi
    
    echo -e "${GREEN}kCompanion system uninstalled.${NC}"
}

# Function to show status
show_status() {
    echo -e "${YELLOW}kCompanion System Status:${NC}"
    
    # Check kernel module
    if lsmod | grep -q "kcompanion"; then
        echo -e "${GREEN}Kernel module: Loaded${NC}"
    else
        echo -e "${RED}Kernel module: Not loaded${NC}"
    fi
    
    # Check device file
    if [ -c "/dev/kcompanion" ]; then
        echo -e "${GREEN}Device file: Available${NC}"
        ls -l /dev/kcompanion
    else
        echo -e "${RED}Device file: Not available${NC}"
    fi
    
    # Check AI daemon
    if [ -f "${SERVICE_DIR}/kcompanion-ai.service" ]; then
        systemctl status kcompanion-ai --no-pager
    elif [ -f "${KCOMP_DIR}/ai_daemon.pid" ]; then
        PID=$(cat "${KCOMP_DIR}/ai_daemon.pid")
        if ps -p $PID > /dev/null; then
            echo -e "${GREEN}AI daemon: Running (PID: $PID)${NC}"
        else
            echo -e "${RED}AI daemon: Not running (stale PID file)${NC}"
        fi
    else
        echo -e "${RED}AI daemon: Not running${NC}"
    fi
    
    # Show suggestions if available
    if [ -c "/dev/kcompanion" ]; then
        echo -e "${YELLOW}\nCurrent Suggestions:${NC}"
        cat /dev/kcompanion 2>/dev/null || echo "No suggestions available."
    fi
}

# Display help
show_help() {
    echo -e "${BLUE}kCompanion AI System Setup Script${NC}"
    echo -e "${YELLOW}Usage:${NC} $0 [command]"
    echo
    echo -e "${YELLOW}Commands:${NC}"
    echo "  install    Check requirements and install the system"
    echo "  start      Start kernel module and AI daemon"
    echo "  stop       Stop AI daemon and unload kernel module"
    echo "  status     Show system status"
    echo "  uninstall  Remove the system"
    echo "  help       Show this help message"
}

# Main script logic
case "$1" in
    install)
        install_system
        ;;
    start)
        start_system
        ;;
    stop)
        stop_system
        ;;
    status)
        show_status
        ;;
    uninstall)
        uninstall_system
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac

exit 0
