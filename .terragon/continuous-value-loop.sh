#!/bin/bash
# Terragon Continuous Value Discovery Loop
# Executes perpetual value discovery and improvement cycles

set -e

REPO_ROOT="$(pwd)"
TERRAGON_DIR="$REPO_ROOT/.terragon"
LOG_FILE="$TERRAGON_DIR/continuous-execution.log"

# Ensure terragon directory exists
mkdir -p "$TERRAGON_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Initialize continuous value discovery
init_continuous_discovery() {
    log "🚀 Initializing Terragon Continuous Value Discovery"
    log "Repository: $(basename "$REPO_ROOT")"
    log "Working directory: $REPO_ROOT"
    
    # Check if git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log "❌ Not a git repository. Continuous discovery requires git."
        exit 1
    fi
    
    # Check for required files
    if [[ ! -f "$TERRAGON_DIR/simple-discovery-engine.py" ]]; then
        log "❌ Simple discovery engine not found. Please run setup first."
        exit 1
    fi
    
    log "✅ Initialization complete"
}

# Run discovery cycle
run_discovery_cycle() {
    log "🔍 Running value discovery cycle"
    
    if python3 "$TERRAGON_DIR/simple-discovery-engine.py" >> "$LOG_FILE" 2>&1; then
        log "✅ Discovery cycle completed successfully"
        
        # Check if backlog was generated
        if [[ -f "BACKLOG.md" ]]; then
            local item_count=$(grep -c "^|" BACKLOG.md | tail -1 || echo "0")
            log "📊 Generated backlog with priority items"
        fi
        
        return 0
    else
        log "❌ Discovery cycle failed"
        return 1
    fi
}

# Check for high-priority items ready for autonomous execution
check_executable_items() {
    if [[ ! -f "$TERRAGON_DIR/value-metrics.json" ]]; then
        return 1
    fi
    
    # Simple check for high-score items (>= 80)
    if python3 -c "
import json
try:
    with open('$TERRAGON_DIR/value-metrics.json') as f:
        data = json.load(f)
    print('executable' if data.get('topScore', 0) >= 80 else 'none')
except:
    print('none')
" | grep -q "executable"; then
        return 0
    else
        return 1
    fi
}

# Execute autonomous improvements (simplified)
execute_autonomous_improvements() {
    log "🤖 Checking for autonomous execution opportunities"
    
    if check_executable_items; then
        log "⚡ High-value items detected - would execute in full autonomous mode"
        log "   (Autonomous execution available with full Terragon system)"
        
        # In this simplified version, we log the opportunity
        log "📝 Recommended next actions:"
        if [[ -f "BACKLOG.md" ]]; then
            local next_item=$(grep "Next Best Value Item" -A3 BACKLOG.md | grep "^\*\*" | head -1)
            log "   - ${next_item//**\[/}"
        fi
    else
        log "📊 No high-priority items ready for autonomous execution"
    fi
}

# Continuous loop with adaptive intervals
run_continuous_loop() {
    local cycle_count=0
    local last_discovery=0
    local discovery_interval=3600  # 1 hour base interval
    
    log "🔄 Starting continuous value discovery loop"
    log "   Discovery interval: ${discovery_interval}s (adaptive)"
    
    while true; do
        cycle_count=$((cycle_count + 1))
        current_time=$(date +%s)
        
        log "📊 Cycle #$cycle_count started"
        
        # Run discovery if interval elapsed
        if (( current_time - last_discovery >= discovery_interval )); then
            if run_discovery_cycle; then
                last_discovery=$current_time
                
                # Check for autonomous execution opportunities
                execute_autonomous_improvements
                
                # Adaptive interval based on findings
                if check_executable_items; then
                    discovery_interval=1800  # 30 minutes if high-value items found
                    log "⚡ High-value items found - increasing discovery frequency"
                else
                    discovery_interval=7200  # 2 hours if low activity
                    log "📊 Low-priority items - reducing discovery frequency"
                fi
            else
                log "❌ Discovery cycle failed - extending interval"
                discovery_interval=14400  # 4 hours on failure
            fi
        fi
        
        # Lightweight monitoring check every 5 minutes
        log "💤 Sleeping for 300s (monitoring mode)"
        sleep 300
        
        # Health check - ensure repository is still valid
        if ! git rev-parse --git-dir > /dev/null 2>&1; then
            log "❌ Repository no longer valid - stopping continuous loop"
            break
        fi
    done
}

# Signal handlers for graceful shutdown
cleanup() {
    log "🛑 Received shutdown signal - cleaning up"
    log "📊 Continuous discovery stopped gracefully"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main execution
main() {
    case "${1:-continuous}" in
        "init")
            init_continuous_discovery
            ;;
        "discovery")
            init_continuous_discovery
            run_discovery_cycle
            ;;
        "once")
            init_continuous_discovery
            run_discovery_cycle
            execute_autonomous_improvements
            ;;
        "continuous")
            init_continuous_discovery
            run_continuous_loop
            ;;
        "status")
            if [[ -f "$LOG_FILE" ]]; then
                log "📊 Continuous Discovery Status"
                log "Last 10 log entries:"
                tail -10 "$LOG_FILE"
            else
                log "❌ No continuous discovery running"
            fi
            ;;
        *)
            echo "Usage: $0 [init|discovery|once|continuous|status]"
            echo "  init       - Initialize discovery system"
            echo "  discovery  - Run single discovery cycle"
            echo "  once       - Run discovery + check for execution"
            echo "  continuous - Start continuous loop (default)"
            echo "  status     - Show current status"
            exit 1
            ;;
    esac
}

main "$@"