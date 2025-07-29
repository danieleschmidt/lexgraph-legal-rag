# GitHub Workflows Implementation Guide

## Quick Setup Checklist

- [ ] Copy workflow files from `docs/github-workflows-setup/*.yml` to `.github/workflows/`
- [ ] Configure required repository secrets
- [ ] Update team references in CODEOWNERS
- [ ] Enable Dependabot security updates
- [ ] Install pre-commit hooks locally
- [ ] Test workflows with a pull request

## Repository Secrets Configuration

Go to **Settings → Secrets and variables → Actions** and add:

### Required Secrets
```bash
CODECOV_TOKEN          # From codecov.io for coverage reporting
```

### Optional Secrets
```bash
SLACK_WEBHOOK_URL      # For release notifications
PYPI_TOKEN            # For automated PyPI publishing
```

## Team Configuration

Update the following teams in your GitHub organization:
- `@terragon-labs/maintainers`
- `@terragon-labs/core-developers`
- `@terragon-labs/security-team`
- `@terragon-labs/devops-team` 
- `@terragon-labs/qa-team`

Or modify `CODEOWNERS` to match your team structure.

## Workflow Features Summary

| Workflow | Trigger | Duration | Purpose |
|----------|---------|----------|---------|
| CI | Every PR/Push | ~5 min | Testing, linting, security |
| Security | Weekly + PR | ~3 min | Vulnerability scanning |
| Performance | Main + PR | ~10 min | Load/stress testing |
| SBOM | Releases | ~2 min | Supply chain security |
| Mutation | Weekly | ~30 min | Test quality assessment |
| Release | Main only | ~8 min | Automated releases |
| Dependencies | Weekly | ~2 min | Dependency updates |

## Pre-commit Setup

For local development:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files (optional)
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

1. **Workflow permission errors**
   - Ensure GitHub Actions has read/write permissions
   - Check repository settings → Actions → General

2. **Codecov token missing**
   - Create account at codecov.io
   - Add repository and copy token to secrets

3. **Team mentions failing**
   - Update CODEOWNERS with actual team names
   - Ensure teams exist in your organization

4. **Docker build failures**
   - Verify Dockerfile exists and is valid
   - Check base image availability

### Testing Workflows

1. Create a test branch
2. Make a small change
3. Open a pull request
4. Verify all workflows execute successfully
5. Check workflow logs for any errors

## Monitoring and Maintenance

- **Weekly**: Review failed workflow runs
- **Monthly**: Update workflow action versions
- **Quarterly**: Review and optimize workflow performance
- **As needed**: Add new workflows for emerging requirements

## Security Considerations

- All workflows follow security best practices
- Secrets are properly scoped and encrypted
- No secrets are logged or exposed
- SBOM generation ensures supply chain transparency
- Container scanning prevents vulnerable deployments

This implementation provides enterprise-grade SDLC capabilities while maintaining developer productivity and code quality.