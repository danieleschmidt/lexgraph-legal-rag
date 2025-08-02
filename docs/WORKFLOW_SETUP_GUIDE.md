# GitHub Workflows Setup Guide

This comprehensive guide provides step-by-step instructions for setting up GitHub Actions workflows for the LexGraph Legal RAG project. Since the Terragon agent cannot directly create workflow files due to GitHub App permissions, this guide enables repository maintainers to manually configure CI/CD pipelines.

## Overview

The LexGraph Legal RAG project uses a comprehensive CI/CD strategy with the following workflows:
- **Continuous Integration (CI)**: Code quality, testing, and validation
- **Continuous Deployment (CD)**: Automated deployment to staging and production
- **Security Scanning**: Vulnerability detection and dependency auditing
- **Dependency Management**: Automated dependency updates
- **Release Management**: Semantic versioning and release automation

## Prerequisites

### Repository Configuration

1. **Enable GitHub Actions**
   - Go to repository Settings → Actions → General
   - Select "Allow all actions and reusable workflows"
   - Enable "Allow GitHub Actions to create and approve pull requests"

2. **Branch Protection Rules**
   ```bash
   # Configure via GitHub Settings → Branches → Add rule
   Branch name pattern: main
   
   Required checks:
   ✓ Require status checks to pass before merging
   ✓ Require branches to be up to date before merging
   ✓ Require conversation resolution before merging
   
   Status checks:
   - ci-test (from CI workflow)
   - security-scan (from security workflow)
   - build (from CI workflow)
   ```

3. **Repository Secrets**
   Navigate to Settings → Secrets and variables → Actions and add:
   ```bash
   # Required secrets
   PYPI_API_TOKEN=pypi-your-token-here
   DOCKER_HUB_USERNAME=your-dockerhub-username
   DOCKER_HUB_TOKEN=your-dockerhub-access-token
   CODECOV_TOKEN=your-codecov-token
   
   # Optional secrets for enhanced features
   SLACK_WEBHOOK_URL=your-slack-webhook-for-notifications
   SONAR_TOKEN=your-sonarcloud-token
   ```

## Workflow Files Setup

### Step 1: Create Workflow Directory

```bash
# Create the .github/workflows directory
mkdir -p .github/workflows
```

### Step 2: Deploy Workflow Files

Copy the following workflow files from `docs/workflows/` to `.github/workflows/`:

#### 1. Continuous Integration (`ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.11'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e ".[dev]"
        
    - name: Lint with ruff
      run: ruff check .
      
    - name: Format check with black
      run: black --check .
      
    - name: Type check with mypy
      run: mypy src/
      
    - name: Run tests
      run: pytest tests/ --cov=lexgraph_legal_rag --cov-report=xml
      
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Build Docker image
      run: |
        docker build -t lexgraph-legal-rag:${{ github.sha }} .
        docker build --target production -t lexgraph-legal-rag:prod-${{ github.sha }} .
```

#### 2. Security Scanning (`security.yml`)

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly scan on Mondays

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety pip-audit
        
    - name: Run Bandit security scan
      run: bandit -r src/ -f json -o bandit-report.json
      
    - name: Run Safety check
      run: safety check --json --output safety-report.json
      
    - name: Run pip-audit
      run: pip-audit --format=json --output=audit-report.json
      
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
          audit-report.json

  codeql-analysis:
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
      
    steps:
    - uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python
        config-file: docs/workflows/codeql-config.yml
        
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
      
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
```

#### 3. Continuous Deployment (`cd.yml`)

```yaml
name: CD Pipeline

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  deploy-staging:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        target: production
        push: true
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:staging
        
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add your staging deployment commands here

  deploy-production:
    if: startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        target: production
        push: true
        tags: |
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.ref_name }}
```

#### 4. Dependency Updates (`dependency-update.yml`)

```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Mondays at 6 AM UTC
  workflow_dispatch:

jobs:
  update-dependencies:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install pip-tools
      run: pip install pip-tools
      
    - name: Update dependencies
      run: |
        pip-compile --upgrade requirements.in
        pip-compile --upgrade requirements-dev.in
        
    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'chore(deps): update dependencies'
        title: 'chore(deps): update dependencies'
        body: |
          Automated dependency update
          
          - Updated all dependencies to latest versions
          - Please review changes before merging
        branch: dependency-updates
        delete-branch: true
```

#### 5. Release Management (`release.yml`)

```yaml
name: Release Management

on:
  push:
    branches: [ main ]

jobs:
  release:
    runs-on: ubuntu-latest
    if: "!contains(github.event.head_commit.message, 'skip ci')"
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        token: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
        
    - name: Install semantic-release
      run: npm install -g semantic-release @semantic-release/changelog @semantic-release/git
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        
    - name: Run semantic-release
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        PYPI_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
      run: npx semantic-release
```

### Step 3: Configure CodeQL

Create `docs/workflows/codeql-config.yml`:

```yaml
name: "LexGraph Legal RAG CodeQL Config"

disable-default-queries: false

queries:
  - name: security-extended
    uses: security-extended
  - name: security-and-quality
    uses: security-and-quality

paths-ignore:
  - tests/
  - docs/
  - scripts/

paths:
  - src/
```

## Advanced Configuration

### Environment-Specific Deployments

#### Staging Environment
```yaml
# In repository Settings → Environments
Name: staging
Protection rules:
- No required reviewers
- Wait timer: 0 minutes
Environment secrets:
- STAGING_API_URL
- STAGING_DATABASE_URL
```

#### Production Environment
```yaml
# In repository Settings → Environments  
Name: production
Protection rules:
- Required reviewers: 2
- Wait timer: 5 minutes
Environment secrets:
- PRODUCTION_API_URL
- PRODUCTION_DATABASE_URL
```

### Notification Setup

#### Slack Integration
```yaml
# Add to workflow files for notifications
- name: Slack Notification
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#deployments'
    webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### Performance Monitoring Integration

#### Add to CI workflow:
```yaml
- name: Performance Testing
  run: |
    # Start application
    docker-compose up -d
    sleep 30
    
    # Run k6 performance tests
    k6 run tests/performance/load-test.js
    
    # Cleanup
    docker-compose down
```

## Monitoring and Maintenance

### Workflow Health Monitoring

1. **Workflow Status Dashboard**
   - Monitor in GitHub Actions tab
   - Set up alerts for failed workflows
   - Review workflow run times regularly

2. **Dependency Update Monitoring**
   - Review automated dependency PRs weekly
   - Monitor for security vulnerabilities
   - Update workflow actions regularly

3. **Performance Metrics**
   - Track CI/CD pipeline execution times
   - Monitor build success rates
   - Optimize slow-running jobs

### Troubleshooting Common Issues

#### Failed CI Tests
```bash
# Debug failed tests locally
docker run --rm -v $(pwd):/app -w /app python:3.11 bash -c "
  pip install -r requirements.txt &&
  pip install -e . &&
  pytest tests/ -v
"
```

#### Docker Build Failures
```bash
# Test Docker build locally
docker build --target development -t lexgraph-dev .
docker build --target production -t lexgraph-prod .
```

#### Dependency Conflicts
```bash
# Resolve dependency issues
pip-compile --upgrade --resolver=backtracking requirements.in
```

## Security Best Practices

### Secrets Management
- Use GitHub Secrets for sensitive data
- Rotate secrets regularly
- Use environment-specific secrets
- Never commit secrets to repository

### Action Security
- Pin action versions to specific commits
- Review third-party actions before use
- Use official actions when possible
- Enable Dependabot for workflow updates

### Code Security
- Enable branch protection rules
- Require code review for changes
- Use CodeQL for security analysis
- Regular security scanning

## Validation Checklist

After setting up workflows, verify:

- [ ] All workflow files are in `.github/workflows/`
- [ ] Required secrets are configured
- [ ] Branch protection rules are enabled
- [ ] Environments are configured
- [ ] Test workflows run successfully
- [ ] Notifications are working
- [ ] Security scans are enabled
- [ ] Release automation works

## Next Steps

1. **Deploy workflow files** to `.github/workflows/`
2. **Configure secrets** in repository settings
3. **Set up environments** for staging and production
4. **Enable branch protection** rules
5. **Test workflows** with a sample PR
6. **Monitor and optimize** workflow performance

This comprehensive setup ensures robust CI/CD pipelines for the LexGraph Legal RAG project with proper security, testing, and deployment automation.