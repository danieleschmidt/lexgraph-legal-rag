#!/bin/bash
set -e

echo "🚀 Setting up LexGraph Legal RAG development environment..."

# Update package lists
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    jq \
    htop \
    vim \
    tree \
    unzip

# Install k6 for performance testing
echo "📊 Installing k6 for performance testing..."
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# Create and activate virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install \
    black \
    ruff \
    mypy \
    bandit \
    safety \
    pre-commit \
    pytest-xdist \
    pytest-benchmark \
    mutmut \
    pip-audit \
    cyclonedx-bom \
    pip-licenses

# Install additional tools
pip install \
    ipython \
    jupyter \
    rich \
    typer \
    httpie

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env 2>/dev/null || echo "# Development environment variables" > .env
    echo "API_KEY=dev-secret-key" >> .env
    echo "ENVIRONMENT=development" >> .env
    echo "LOG_LEVEL=DEBUG" >> .env
fi

# Set up git configuration for container
git config --global --add safe.directory /workspaces/lexgraph-legal-rag

# Create useful aliases
echo "🔧 Setting up useful aliases..."
cat >> ~/.bashrc << 'EOF'

# LexGraph Development Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Project-specific aliases
alias activate='source /opt/venv/bin/activate'
alias test='pytest tests/'
alias test-fast='pytest tests/ -x --ff'
alias test-cov='pytest tests/ --cov=lexgraph_legal_rag --cov-report=html'
alias lint='ruff check src/ tests/'
alias format='black src/ tests/'
alias typecheck='mypy src/'
alias security='bandit -r src/'
alias dev='python -m uvicorn lexgraph_legal_rag.api:create_api --host 0.0.0.0 --port 8000 --reload'
alias build='python -m build'
alias clean='find . -type f -name "*.pyc" -delete && find . -type d -name "__pycache__" -delete'

# Git aliases
alias gst='git status'
alias gco='git checkout'
alias glog='git log --oneline --graph --decorate'
alias gpull='git pull origin'
alias gpush='git push origin'

# Docker aliases
alias dps='docker ps'
alias dimg='docker images'
alias dlog='docker logs -f'
alias dcup='docker-compose up -d'
alias dcdown='docker-compose down'

# Kubernetes aliases (if using k8s)
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgs='kubectl get services'
alias kdp='kubectl describe pod'
alias klog='kubectl logs -f'
EOF

# Create project structure if missing
mkdir -p {docs,tests,scripts,data,logs}

# Set permissions
chmod +x .devcontainer/setup.sh
find scripts/ -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true

echo "✅ Development environment setup complete!"
echo ""
echo "🎉 Ready for development! Here are some useful commands:"
echo ""
echo "  📋 Run tests:           test"
echo "  🚀 Start dev server:    dev"
echo "  🔧 Format code:         format"
echo "  🔍 Lint code:           lint"
echo "  🛡️  Security scan:       security"
echo "  📊 Type check:          typecheck"
echo "  🧪 Coverage report:     test-cov"
echo ""
echo "Happy coding! 🚀"