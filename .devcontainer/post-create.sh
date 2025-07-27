#!/bin/bash

# DevContainer post-create setup script for LexGraph Legal RAG

set -e

echo "🚀 Starting LexGraph Legal RAG development environment setup..."

# Upgrade pip and install build tools
python -m pip install --upgrade pip setuptools wheel

# Install the project in development mode
echo "📦 Installing project dependencies..."
pip install -e .

# Install development dependencies
pip install -r requirements.txt

# Install additional development tools
pip install \
    pytest-cov \
    pytest-mock \
    pytest-asyncio \
    black \
    ruff \
    mypy \
    bandit \
    safety \
    pre-commit \
    jupyter \
    ipython

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️  Please update .env file with your actual API keys and configuration"
fi

# Create necessary directories
mkdir -p {data,logs,models,indices}

# Set up Git configuration for the container
echo "🔧 Configuring Git..."
git config --global init.defaultBranch main
git config --global pull.rebase false

# Install kubectl completion
echo "🔧 Setting up kubectl completion..."
kubectl completion bash >> ~/.bashrc

# Create useful aliases
echo "🔧 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# LexGraph Legal RAG development aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Python aliases
alias py='python'
alias pip='python -m pip'
alias pytest='python -m pytest'

# Development workflow aliases
alias test='python -m pytest tests/ -v'
alias test-cov='python -m pytest tests/ --cov=lexgraph_legal_rag --cov-report=html'
alias lint='ruff check . && black --check .'
alias format='black . && ruff check --fix .'
alias security='bandit -r src/ && safety check'

# Docker aliases
alias dc='docker-compose'
alias dcu='docker-compose up'
alias dcd='docker-compose down'
alias dcb='docker-compose build'

# Kubernetes aliases
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgs='kubectl get services'
alias kgd='kubectl get deployments'

# Monitoring aliases
alias metrics='curl http://localhost:8001/metrics'
alias health='curl http://localhost:8000/health'
EOF

# Make the script executable
chmod +x .devcontainer/post-create.sh

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  make test          # Run test suite"
echo "  make dev           # Start development server"
echo "  make docker-up     # Start with Docker Compose"
echo "  make lint          # Run code quality checks"
echo ""
echo "📚 Documentation:"
echo "  README.md          # Project overview"
echo "  DEVELOPMENT.md     # Development guide"
echo "  ARCHITECTURE.md    # System architecture"
echo ""
echo "🔧 VS Code is configured with:"
echo "  - Python formatting (Black)"
echo "  - Linting (Ruff)"
echo "  - Testing (pytest)"
echo "  - Type checking (mypy)"
echo "  - Git integration (GitLens)"
echo ""