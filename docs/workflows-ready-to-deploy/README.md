# Advanced Workflow Templates - Ready to Deploy

## Overview

This directory contains **enterprise-grade GitHub workflow templates** that implement advanced SDLC optimization for repositories with ADVANCED (75%+) maturity level.

## Workflow Files

### 1. `advanced-performance.yml`
**Comprehensive Performance Testing and Optimization**
- Multi-platform benchmarking with pytest-benchmark
- Load testing automation with k6
- Memory profiling with py-spy and memory-profiler
- Performance regression detection for PRs
- Advanced performance metrics collection

### 2. `advanced-security.yml` 
**Multi-layered Security Scanning and Compliance**
- Static analysis with Bandit, Semgrep, and PyLint
- Dependency vulnerability scanning with Safety and pip-audit
- Container security scanning with Trivy and Docker Scout
- SBOM generation and validation
- Automated compliance reporting

### 3. `intelligent-deployment.yml`
**Smart Deployment Pipeline with Multiple Strategies**
- Blue-green, canary, and rolling deployment options
- Automated rollback with failure detection
- Pre-deployment validation and health monitoring
- Environment-specific deployment workflows
- Production monitoring integration

## Deployment Instructions

### Prerequisites
1. **Repository Permissions**: Ensure you have admin access to enable workflows
2. **Secrets Configuration**: Set up required secrets (see below)
3. **Branch Protection**: Configure branch protection rules for main/develop

### Step 1: Copy Workflow Files
```bash
# Copy workflow files to .github/workflows/
cp docs/workflows-ready-to-deploy/*.yml .github/workflows/

# Commit the workflow files
git add .github/workflows/
git commit -m "feat: add advanced SDLC workflows"
git push
```

### Step 2: Configure Repository Secrets

#### Required Secrets
```bash
# GitHub token with workflows permissions
GITHUB_TOKEN  # Automatically provided by GitHub

# Container registry credentials (if using container deployment)
REGISTRY_USERNAME  # Container registry username
REGISTRY_PASSWORD  # Container registry password

# Deployment credentials (environment-specific)
STAGING_DEPLOY_KEY    # Staging deployment credentials
PRODUCTION_DEPLOY_KEY # Production deployment credentials

# Monitoring and observability (optional)
PROMETHEUS_URL        # Prometheus monitoring endpoint
GRAFANA_API_KEY      # Grafana dashboard API key
```

#### Optional Secrets for Advanced Features
```bash
# Security scanning enhancements
SNYK_TOKEN           # Snyk vulnerability scanning
SONARCLOUD_TOKEN     # SonarCloud code quality

# Performance monitoring
NEW_RELIC_API_KEY    # New Relic performance monitoring
DATADOG_API_KEY      # Datadog observability

# Deployment platforms
KUBERNETES_CONFIG    # Kubernetes cluster configuration
AWS_ACCESS_KEY_ID    # AWS deployment credentials
AWS_SECRET_ACCESS_KEY
```

### Step 3: Enable Workflows
1. Navigate to **Settings > Actions > General**
2. Set **Actions permissions** to "Allow all actions and reusable workflows"
3. Enable **Workflow permissions** to "Read and write permissions"
4. Save settings

### Step 4: Configure Branch Protection
```bash
# Enable branch protection for main branch
# Settings > Branches > Add rule
Branch name pattern: main
✅ Require status checks to pass before merging
✅ Require branches to be up to date before merging
✅ Require review from code owners
✅ Restrict pushes that create files in .github/workflows/
```

## Workflow Configuration

### Performance Testing Configuration
Edit `advanced-performance.yml` to customize:
```yaml
env:
  PERFORMANCE_THRESHOLD: '5'  # Adjust performance regression threshold
  PYTHON_VERSION: '3.11'     # Set Python version for testing
```

### Security Scanning Configuration  
Edit `advanced-security.yml` to customize:
```yaml
# Adjust security scanning frequency
schedule:
  - cron: '0 1 * * 2'  # Weekly Tuesday 1 AM UTC

# Configure security tools
SECURITY_THRESHOLD: 'medium'  # minimum vulnerability level
```

### Deployment Configuration
Edit `intelligent-deployment.yml` to customize:
```yaml
# Set default deployment strategy
env:
  DEFAULT_STRATEGY: 'rolling'  # or 'blue-green', 'canary'
  
# Configure environments
environments:
  staging: 'staging'
  production: 'production'
```

## Usage Examples

### Triggering Performance Tests
```bash
# Automatically triggered on PRs with 'performance' label
gh pr create --label performance

# Manual trigger with specific benchmark type
gh workflow run advanced-performance.yml -f benchmark_type=memory
```

### Security Scanning
```bash
# Comprehensive security scan
gh workflow run advanced-security.yml -f scan_type=comprehensive

# Dependency-only scan
gh workflow run advanced-security.yml -f scan_type=dependency-only
```

### Deployment Options
```bash
# Rolling deployment to staging
gh workflow run intelligent-deployment.yml -f deployment_strategy=rolling -f environment=staging

# Canary deployment to production
gh workflow run intelligent-deployment.yml -f deployment_strategy=canary -f environment=production
```

## Monitoring and Observability

### Performance Metrics
- **Benchmark Results**: Stored as GitHub artifacts
- **Performance Trends**: Tracked via GitHub Actions metrics
- **Regression Alerts**: Automated PR comments for performance degradation

### Security Metrics
- **Vulnerability Reports**: Uploaded to GitHub Security tab
- **Compliance Status**: Tracked in security dashboard
- **SBOM Artifacts**: Available for supply chain analysis

### Deployment Metrics
- **Deployment Success Rate**: Tracked across environments
- **Rollback Frequency**: Monitored for process improvement
- **Health Check Results**: Real-time deployment validation

## Customization Guide

### Adding Custom Performance Tests
1. Create test files in `tests/performance/`
2. Use `pytest-benchmark` for benchmarking
3. Update workflow matrix to include new test categories

### Extending Security Scanning
1. Add new security tools to workflow
2. Configure tool-specific secrets
3. Update SARIF upload for GitHub Security integration

### Custom Deployment Strategies
1. Add new strategy to deployment workflow
2. Implement strategy-specific deployment logic
3. Update documentation and usage examples

## Troubleshooting

### Common Issues

#### Workflow Permission Errors
```bash
# Error: refusing to allow GitHub App to create workflow
# Solution: Manual workflow file creation required
cp docs/workflows-ready-to-deploy/*.yml .github/workflows/
```

#### Performance Test Failures
```bash
# Check performance threshold settings
# Review baseline benchmark results
# Validate test environment setup
```

#### Security Scan False Positives
```bash
# Configure security tool exclusions
# Update security baseline files
# Review security policy settings
```

#### Deployment Failures
```bash
# Verify deployment credentials
# Check environment configuration
# Review health check endpoints
```

## Support and Maintenance

### Regular Maintenance
- **Weekly**: Review security scan results
- **Monthly**: Update performance baselines
- **Quarterly**: Review and optimize workflow performance

### Updates and Improvements
- Monitor workflow execution times
- Update tool versions and configurations
- Gather team feedback for process improvements

### Best Practices
1. **Test workflows in staging** before production deployment
2. **Monitor workflow costs** and optimize resource usage
3. **Keep secrets updated** and rotated regularly
4. **Document customizations** for team knowledge sharing
5. **Regular training** on new workflow features

## Conclusion

These advanced workflows provide **enterprise-grade SDLC automation** with:
- **Intelligent performance monitoring** and regression prevention
- **Comprehensive security scanning** and compliance automation
- **Smart deployment strategies** with automated rollback
- **Modern development practices** and tool integration

The workflows are designed to be **production-ready** while remaining **customizable** for specific team and project needs.

---
*Generated by Terragon Autonomous SDLC Enhancement*  
*Ready for immediate deployment in advanced repositories*