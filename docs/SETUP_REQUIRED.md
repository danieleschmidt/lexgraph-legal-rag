# Manual Setup Requirements

This document outlines the manual setup steps required to complete the SDLC implementation for the LexGraph Legal RAG project. These steps cannot be automated due to GitHub App permission limitations.

## Required Manual Actions

### 1. GitHub Workflows Setup (HIGH PRIORITY)

#### Action Required
Copy workflow files from `docs/workflows/` to `.github/workflows/` directory:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow files
cp docs/workflows/ci.yml .github/workflows/
cp docs/workflows/cd.yml .github/workflows/
cp docs/workflows/security.yml .github/workflows/
cp docs/workflows/dependency-update.yml .github/workflows/
cp docs/workflows/release.yml .github/workflows/
cp docs/workflows/codeql-config.yml .github/workflows/
```

#### Configuration Required
1. **Repository Secrets** (Settings → Secrets and variables → Actions):
   ```
   PYPI_API_TOKEN=your-pypi-token
   DOCKER_HUB_USERNAME=your-dockerhub-username
   DOCKER_HUB_TOKEN=your-dockerhub-token
   CODECOV_TOKEN=your-codecov-token
   SLACK_WEBHOOK_URL=your-slack-webhook (optional)
   ```

2. **Branch Protection Rules** (Settings → Branches):
   - Protect `main` branch
   - Require status checks: ci-test, security-scan, build
   - Require pull request reviews
   - Require conversation resolution

3. **Environments** (Settings → Environments):
   - Create `staging` environment (no protection rules)
   - Create `production` environment (require 2 reviewers, 5-minute timer)

### 2. Repository Settings Configuration

#### Security Settings
- Enable vulnerability alerts
- Enable Dependabot security updates
- Enable Dependabot version updates
- Configure CodeQL code scanning

#### General Settings
- Enable Issues and Pull Requests
- Enable Discussions (optional)
- Set repository description: "LangGraph-powered multi-agent legal document analysis system"
- Add topics: `legal`, `rag`, `ai`, `langgraph`, `multi-agent`, `python`, `fastapi`

### 3. Required Integrations

#### Code Coverage (Codecov)
1. Sign up at https://codecov.io
2. Connect your GitHub repository
3. Copy the Codecov token to repository secrets

#### Container Registry
1. Set up GitHub Container Registry or Docker Hub
2. Configure authentication tokens
3. Update workflow files with correct registry URLs

### 4. Documentation Website (Optional)

#### GitHub Pages Setup
1. Go to Settings → Pages
2. Select source: Deploy from a branch
3. Choose branch: `main`, folder: `/docs`
4. Custom domain (optional): `docs.yourproject.com`

### 5. Project Management Setup

#### GitHub Project Board
1. Create new project (Settings → Projects)
2. Import issues from `BACKLOG.md`
3. Set up automation rules for issue lifecycle

#### Labels Configuration
Create repository labels:
- `bug` (red) - Something isn't working
- `enhancement` (green) - New feature or request
- `documentation` (blue) - Improvements or additions to documentation
- `security` (red) - Security-related issues
- `performance` (orange) - Performance improvements
- `testing` (yellow) - Testing-related changes

### 6. Quality Gates Configuration

#### Minimum Requirements
Configure branch protection to require:
- 80% test coverage minimum
- All security scans pass
- All linting checks pass
- At least 1 code review approval

#### Performance Thresholds
Set up monitoring alerts for:
- API response time > 1 second (95th percentile)
- Error rate > 1% (5-minute average)
- Memory usage > 2GB
- CPU usage > 80%

## Verification Steps

After completing manual setup, verify:

### 1. Workflow Functionality
```bash
# Create a test branch and PR to verify workflows
git checkout -b test-workflows
echo "# Test" > test.md
git add test.md
git commit -m "test: verify workflow functionality"
git push -u origin test-workflows
# Create PR via GitHub UI
```

### 2. Security Scanning
- Check that CodeQL scans run successfully
- Verify Dependabot alerts are enabled
- Confirm secret scanning is active

### 3. Build and Deployment
- Verify Docker images build successfully
- Test deployment to staging environment
- Confirm monitoring and alerting work

### 4. Documentation Access
- Verify documentation is accessible
- Test all internal links
- Confirm API documentation generates correctly

## Permissions Required

The following GitHub permissions are needed for full functionality:

### Repository Permissions
- **Contents**: Read and write (for code and documentation)
- **Issues**: Write (for automated issue creation)
- **Pull requests**: Write (for automated PRs)
- **Actions**: Write (for workflow management)
- **Security events**: Write (for security scanning)
- **Pages**: Write (for documentation hosting)

### Organization Permissions
- **Members**: Read (for team assignments)
- **Projects**: Write (for project board integration)

## Timeline for Manual Setup

### Immediate (Day 1)
- [ ] Copy workflow files to `.github/workflows/`
- [ ] Configure repository secrets
- [ ] Set up branch protection rules
- [ ] Enable security features

### Within 1 Week
- [ ] Set up monitoring integrations
- [ ] Configure deployment environments
- [ ] Test full CI/CD pipeline
- [ ] Verify all documentation links

### Ongoing
- [ ] Monitor workflow performance
- [ ] Review and update security settings
- [ ] Optimize build and deployment times
- [ ] Update documentation as needed

## Success Criteria

Setup is complete when:
- [ ] All workflows run successfully on PR creation
- [ ] Security scans execute without errors
- [ ] Automated deployment to staging works
- [ ] Monitoring and alerting are functional
- [ ] Documentation is accessible and up-to-date
- [ ] Team can contribute using standard Git workflow

This manual setup ensures the full SDLC implementation is operational and provides a robust development and deployment pipeline for the LexGraph Legal RAG project.