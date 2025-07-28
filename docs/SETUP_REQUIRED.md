# Manual Setup Requirements

## Repository Configuration

### GitHub Actions Setup
1. Navigate to repository Settings → Actions → General
2. Enable "Allow all actions and reusable workflows"
3. Set workflow permissions to "Read and write permissions"

### Branch Protection Rules
Configure for `main` branch:
- Require pull request reviews before merging
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Include administrators in restrictions

### Repository Secrets
Add these secrets in Settings → Secrets and variables → Actions:
```
PYPI_API_TOKEN=<your-pypi-token>
DOCKER_HUB_TOKEN=<your-docker-token>
CODECOV_TOKEN=<your-codecov-token>
```

### Workflow Deployment
Copy workflow files from `docs/workflows/` to `.github/workflows/`:
```bash
cp docs/workflows/*.yml .github/workflows/
git add .github/workflows/
git commit -m "ci: add GitHub Actions workflows"
```

## External Services

### Code Coverage
1. Sign up at [Codecov.io](https://codecov.io)
2. Connect your repository
3. Copy token to repository secrets

### Container Registry
1. Create account at [Docker Hub](https://hub.docker.com)
2. Generate access token
3. Add token to repository secrets

## Verification

After setup, verify:
- [ ] Actions tab shows enabled workflows
- [ ] Branch protection rules are active
- [ ] All required secrets are configured
- [ ] First workflow run completes successfully