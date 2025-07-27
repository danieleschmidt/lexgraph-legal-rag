# GitHub Workflows Setup Guide

This guide explains how to manually set up the comprehensive CI/CD workflows for LexGraph Legal RAG, since GitHub Apps cannot directly create workflow files.

## 🚀 Quick Setup

### Step 1: Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### Step 2: Copy Workflow Files
Copy the following files from `docs/workflows/` to `.github/workflows/`:

- `ci.yml` - Continuous Integration
- `cd.yml` - Continuous Deployment  
- `security.yml` - Security Scanning
- `release.yml` - Release Management
- `dependency-update.yml` - Dependency Management

### Step 3: Create CodeQL Directory
```bash
mkdir -p .github/codeql
cp docs/workflows/codeql-config.yml .github/codeql/
```

### Step 4: Configure Secrets
Add the following secrets in GitHub repository settings:

#### Required Secrets
- `PYPI_API_TOKEN` - For package publishing
- `DOCKERHUB_USERNAME` - Docker Hub username
- `DOCKERHUB_TOKEN` - Docker Hub access token
- `SLACK_WEBHOOK` - For notifications (optional)
- `GITGUARDIAN_API_KEY` - For secret scanning (optional)
- `CODECOV_TOKEN` - For coverage reporting (optional)

## 📋 Workflow Overview

### 1. Continuous Integration (`ci.yml`)
**Triggers**: Push to main/develop, Pull Requests

**Features**:
- Code quality checks (Ruff, Black, MyPy)
- Matrix testing (Python 3.8-3.11)
- Security scanning (Bandit, Safety)
- Build verification
- Docker image testing
- Performance tests (k6)
- Quality gates

**Jobs**:
- `code-quality` - Linting, formatting, type checking
- `test` - Unit and integration tests with coverage
- `security` - Security vulnerability scanning
- `build` - Package building and validation
- `docker` - Docker image building and testing
- `performance` - Performance testing with k6
- `quality-gate` - Final validation

### 2. Continuous Deployment (`cd.yml`)
**Triggers**: Push to main, Release tags

**Features**:
- Staging deployment automation
- Blue-green production deployment
- Package publishing to PyPI
- Multi-platform Docker images
- Post-deployment verification
- Automated rollback capability

**Jobs**:
- `deploy-staging` - Deploy to staging environment
- `deploy-production` - Production deployment (tags only)
- `publish-package` - PyPI package publishing
- `publish-docker` - Multi-platform Docker publishing
- `migrate-database` - Database migrations
- `post-deployment-tests` - Verification tests

### 3. Security Scanning (`security.yml`)
**Triggers**: Push, Pull Requests, Weekly schedule

**Features**:
- Static Application Security Testing (SAST)
- CodeQL analysis
- Dependency vulnerability scanning
- Container security scanning
- Secrets detection
- Infrastructure as Code security
- SBOM generation

**Tools Integrated**:
- Bandit (Python SAST)
- Safety (Dependency scanning)
- Semgrep (Pattern-based scanning)
- CodeQL (GitHub's semantic analysis)
- Trivy (Container/dependency scanning)
- OSV-Scanner (Open source vulnerabilities)
- GitGuardian (Secret detection)
- Checkov (IaC security)

### 4. Release Management (`release.yml`)
**Triggers**: Push to main, Manual workflow dispatch

**Features**:
- Semantic versioning with conventional commits
- Automated changelog generation
- Multi-platform Docker image publishing
- PyPI package publishing
- GitHub release creation
- Documentation updates
- Production deployment

**Workflow**:
1. Calculate next version
2. Run pre-release quality checks
3. Create GitHub release
4. Publish to PyPI
5. Build and push Docker images
6. Deploy to production
7. Update documentation
8. Send notifications

### 5. Dependency Management (`dependency-update.yml`)
**Triggers**: Weekly/Monthly schedule, Manual dispatch

**Features**:
- Automated security updates
- Minor version updates
- Major version tracking
- Dependency health monitoring
- License compliance checking
- Vulnerability scanning

**Update Types**:
- **Security**: Immediate updates for vulnerabilities
- **Minor**: Weekly compatible updates
- **Major**: Monthly tracking with manual review
- **Audit**: Comprehensive dependency health reporting

## 🔧 Configuration

### Branch Protection Rules
Recommended branch protection for `main`:

```yaml
required_status_checks:
  strict: true
  contexts:
    - "Code Quality"
    - "Test Suite"
    - "Security Scan"
    - "Build Package"
    - "Docker Build"
    - "Quality Gate"

enforce_admins: true
required_pull_request_reviews:
  required_approving_review_count: 2
  dismiss_stale_reviews: true
  require_code_owner_reviews: true

restrictions:
  users: []
  teams: ["core-developers"]
```

### Environment Configuration

#### Staging Environment
- **URL**: `https://staging.lexgraph.terragon.ai`
- **Protection**: None (automatic deployment)
- **Secrets**: Staging-specific configuration

#### Production Environment
- **URL**: `https://lexgraph.terragon.ai`
- **Protection**: Required reviewers
- **Secrets**: Production configuration
- **Deployment**: Manual approval required

### Required Repository Variables
```bash
PYTHON_VERSION=3.11
NODE_VERSION=18
REGISTRY=ghcr.io
IMAGE_NAME=lexgraph-legal-rag
```

## 🔍 Monitoring and Alerts

### Workflow Monitoring
- All workflows report status to Slack (if configured)
- Failed workflows create GitHub issues
- Security findings are escalated immediately
- Performance regressions trigger alerts

### Metrics Collection
- Build times and success rates
- Test coverage trends
- Security scan results
- Dependency health scores
- Deployment frequency and success

## 🚨 Troubleshooting

### Common Issues

#### "Permission denied" errors
- Ensure GitHub token has correct permissions
- Check if workflows permission is enabled
- Verify branch protection rules allow automation

#### Failed security scans
- Review security scan reports in workflow artifacts
- Update dependencies to fix vulnerabilities
- Add security exceptions if needed (with justification)

#### Deployment failures
- Check deployment logs in workflow runs
- Verify secrets are correctly configured
- Ensure target environment is accessible

#### Test failures
- Review test reports in workflow artifacts
- Check if environment-specific issues
- Verify test data and fixtures are available

### Getting Help
- Check workflow run logs for detailed error messages
- Review artifact downloads for reports and logs
- Consult individual workflow documentation
- Create issues for persistent problems

## ✅ Verification Checklist

After setting up workflows, verify:

- [ ] All workflow files are in `.github/workflows/`
- [ ] Required secrets are configured
- [ ] Branch protection rules are enabled
- [ ] Staging environment is accessible
- [ ] Production environment is configured
- [ ] Notification channels are working
- [ ] Test workflows with a small PR
- [ ] Security scans complete successfully
- [ ] Build and deployment processes work

## 🔄 Maintenance

### Regular Tasks
- Review security scan results weekly
- Update workflow configurations as needed
- Monitor build performance and optimize
- Update dependencies in workflow files
- Review and adjust quality gates

### Quarterly Reviews
- Assess workflow effectiveness
- Update security scanning tools
- Review branch protection rules
- Optimize build and deployment processes
- Update documentation

---

This comprehensive workflow system provides enterprise-grade CI/CD, security, and automation for LexGraph Legal RAG, ensuring high quality, security, and reliability in production deployments.