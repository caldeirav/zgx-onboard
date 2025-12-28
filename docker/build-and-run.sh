#!/bin/bash
# =============================================================================
# ZGX Nano CUDA Dev Container - Build & Run Script
# =============================================================================
# This script builds and runs the CUDA container for ZGX demos.
#
# Usage:
#   ./build-and-run.sh          # Build and run with default settings
#   ./build-and-run.sh --build  # Only build the image
#   ./build-and-run.sh --run    # Only run (assumes image exists)
#   ./build-and-run.sh --stop   # Stop and remove the container
# =============================================================================

set -e

# Configuration
IMAGE_NAME="zgx-cuda"
CONTAINER_NAME="zgx_cuda"
SHM_SIZE="100g"  # Required for unified memory operations with large models
WORKSPACE_PATH="${WORKSPACE_PATH:-$(pwd)/..}"  # Default to project root

# Named volumes for persisting Cursor/VS Code extensions
CURSOR_EXTENSIONS_VOLUME="zgx-cursor-extensions"
VSCODE_EXTENSIONS_VOLUME="zgx-vscode-extensions"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${GREEN}=== $1 ===${NC}\n"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

build_image() {
    print_header "Building CUDA Dev Container"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    
    echo "Building image: $IMAGE_NAME"
    docker build -t "$IMAGE_NAME" .
    
    print_success "Image built successfully: $IMAGE_NAME"
}

run_container() {
    print_header "Running CUDA Dev Container"
    
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "Container '$CONTAINER_NAME' already exists."
        read -p "Remove and recreate? (y/N): " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            docker rm -f "$CONTAINER_NAME"
        else
            echo "Exiting."
            exit 0
        fi
    fi
    
    # Resolve workspace path
    WORKSPACE_PATH=$(cd "$WORKSPACE_PATH" && pwd)
    
    echo "Configuration:"
    echo "  Container: $CONTAINER_NAME"
    echo "  Image: $IMAGE_NAME"
    echo "  Shared Memory: $SHM_SIZE"
    echo "  Workspace: $WORKSPACE_PATH"
    echo "  Extension volumes: $CURSOR_EXTENSIONS_VOLUME, $VSCODE_EXTENSIONS_VOLUME"
    echo ""
    
    docker run -d \
        --gpus all \
        --shm-size="$SHM_SIZE" \
        -v "$WORKSPACE_PATH":/workspace \
        -v "$CURSOR_EXTENSIONS_VOLUME":/root/.cursor-server \
        -v "$VSCODE_EXTENSIONS_VOLUME":/root/.vscode-server \
        --name "$CONTAINER_NAME" \
        "$IMAGE_NAME"
    
    print_success "Container started: $CONTAINER_NAME"
    echo ""
    echo "Next steps:"
    echo "  1. Open Cursor"
    echo "  2. Press F1 or click the remote indicator (bottom left)"
    echo "  3. Select 'Dev Containers: Attach to Running Container...'"
    echo "  4. Choose '$CONTAINER_NAME'"
    echo "  5. Open notebooks from /workspace/notebooks/"
}

stop_container() {
    print_header "Stopping Dev Container"
    
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker rm -f "$CONTAINER_NAME"
        print_success "Container stopped and removed: $CONTAINER_NAME"
        echo ""
        echo "Note: Extension volumes are preserved. Use --clean to remove them."
    else
        print_warning "Container '$CONTAINER_NAME' not found."
    fi
}

clean_volumes() {
    print_header "Cleaning Extension Volumes"
    
    echo "This will remove all cached Cursor/VS Code extensions."
    read -p "Continue? (y/N): " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        docker volume rm -f "$CURSOR_EXTENSIONS_VOLUME" 2>/dev/null && \
            print_success "Removed: $CURSOR_EXTENSIONS_VOLUME" || \
            print_warning "Volume not found: $CURSOR_EXTENSIONS_VOLUME"
        docker volume rm -f "$VSCODE_EXTENSIONS_VOLUME" 2>/dev/null && \
            print_success "Removed: $VSCODE_EXTENSIONS_VOLUME" || \
            print_warning "Volume not found: $VSCODE_EXTENSIONS_VOLUME"
    else
        echo "Cancelled."
    fi
}

show_status() {
    print_header "Container Status"
    
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_success "Container is running: $CONTAINER_NAME"
        echo ""
        docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "Container exists but is not running: $CONTAINER_NAME"
    else
        echo "Container '$CONTAINER_NAME' does not exist."
    fi
}

show_help() {
    echo "ZGX Nano CUDA Dev Container - Management Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --build    Build the Docker image only"
    echo "  --run      Run the container only (assumes image exists)"
    echo "  --stop     Stop and remove the container (preserves extensions)"
    echo "  --clean    Remove cached Cursor/VS Code extension volumes"
    echo "  --status   Show container status"
    echo "  --help     Show this help message"
    echo ""
    echo "Without options: builds the image and runs the container"
    echo ""
    echo "Environment variables:"
    echo "  WORKSPACE_PATH  Path to mount as /workspace (default: project root)"
    echo ""
    echo "Extension persistence:"
    echo "  Extensions are stored in Docker volumes and persist across container rebuilds."
    echo "  Volumes: $CURSOR_EXTENSIONS_VOLUME, $VSCODE_EXTENSIONS_VOLUME"
}

# Main
case "${1:-}" in
    --build)
        build_image
        ;;
    --run)
        run_container
        ;;
    --stop)
        stop_container
        ;;
    --clean)
        clean_volumes
        ;;
    --status)
        show_status
        ;;
    --help|-h)
        show_help
        ;;
    "")
        build_image
        run_container
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac

