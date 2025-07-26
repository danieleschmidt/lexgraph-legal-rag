#!/bin/bash
set -euo pipefail

# GitHub Repository Hygiene Bot Runner
# Usage: ./run_hygiene_bot.sh [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_SCRIPT="$SCRIPT_DIR/github_hygiene_bot.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [--dry-run]"
    echo ""
    echo "Environment variables required:"
    echo "  GITHUB_TOKEN    - GitHub personal access token with repo permissions"
    echo "  GITHUB_OWNER    - GitHub username or organization name"
    echo ""
    echo "Options:"
    echo "  --dry-run       - Show what would be done without making changes"
    echo "  --help          - Show this help message"
}

log_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Parse arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check required environment variables
if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    log_error "GITHUB_TOKEN environment variable is required"
    echo "Get a token from: https://github.com/settings/tokens"
    echo "Required scopes: repo, read:org, read:user"
    exit 1
fi

if [[ -z "${GITHUB_OWNER:-}" ]]; then
    log_error "GITHUB_OWNER environment variable is required"
    echo "Set to your GitHub username or organization name"
    exit 1
fi

# Check if Python script exists
if [[ ! -f "$BOT_SCRIPT" ]]; then
    log_error "GitHub hygiene bot script not found: $BOT_SCRIPT"
    exit 1
fi

# Check Python dependencies
log_info "Checking Python dependencies..."
if ! python3 -c "import requests" 2>/dev/null; then
    log_error "Python 'requests' library is required"
    echo "Install with: pip install requests"
    exit 1
fi

# Run the bot
log_info "Starting GitHub Repository Hygiene Bot"
log_info "Owner: $GITHUB_OWNER"

if [[ "$DRY_RUN" = true ]]; then
    log_warning "Running in DRY RUN mode - no changes will be made"
    python3 "$BOT_SCRIPT" --token "$GITHUB_TOKEN" --owner "$GITHUB_OWNER" --dry-run
else
    log_info "Running hygiene automation..."
    python3 "$BOT_SCRIPT" --token "$GITHUB_TOKEN" --owner "$GITHUB_OWNER"
fi

log_success "GitHub hygiene automation completed!"