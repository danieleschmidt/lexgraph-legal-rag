# GitHub Repository Hygiene Bot

Automated GitHub repository hygiene maintenance following security and community best practices.

> **Note**: This bot creates GitHub Actions workflow files. If you're using a GitHub App token, ensure it has `workflows` permission, or manually create workflow files in `.github/workflows/`.

## ✨ Features

- **Repository Metadata**: Updates descriptions, homepages, and topics
- **Community Files**: Creates LICENSE, CODE_OF_CONDUCT.md, CONTRIBUTING.md, SECURITY.md
- **Security Scanners**: Enables CodeQL, Dependabot, OSSF Scorecard
- **SBOM Generation**: Creates Software Bill of Materials with vulnerability monitoring
- **README Enhancements**: Adds badges and required sections
- **Repository Management**: Archives stale repos and pins top repositories
- **Metrics Tracking**: Logs hygiene status for monitoring

## ⚡ Quick Start

### Prerequisites

- Python 3.7+
- GitHub personal access token with `repo`, `read:org`, `read:user` permissions
- `requests` library: `pip install requests`

### Setup

1. **Get GitHub Token**:
   ```bash
   # Visit https://github.com/settings/tokens
   # Create token with: repo, read:org, read:user scopes
   export GITHUB_TOKEN="your_token_here"
   ```

2. **Set GitHub Owner**:
   ```bash
   export GITHUB_OWNER="your_username_or_org"
   ```

3. **Run Hygiene Bot**:
   ```bash
   # Dry run to see what would change
   ./run_hygiene_bot.sh --dry-run
   
   # Apply changes
   ./run_hygiene_bot.sh
   ```

### Alternative Python Usage

```bash
python3 github_hygiene_bot.py --token "$GITHUB_TOKEN" --owner "$GITHUB_OWNER"
```

## 🔍 What It Does

### Step 0: Repository Discovery
- Lists all owned repositories
- Filters out forks, templates, and archived repos
- Processes each repository individually

### Step 1: Metadata Updates
- **Description**: Adds description if missing (max 120 chars)
- **Homepage**: Sets to `https://{owner}.github.io` if null  
- **Topics**: Adds relevant topics based on language and patterns

### Step 2: Community Files
Creates missing community files:
- `LICENSE` (Apache 2.0)
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- `CONTRIBUTING.md` (Conventional Commits guide)
- `SECURITY.md` (Vulnerability disclosure policy)
- `.github/ISSUE_TEMPLATE/bug.yml`
- `.github/ISSUE_TEMPLATE/feature.yml`

### Step 3: Security Scanners
Enables security automation:
- **CodeQL**: Static analysis workflow
- **Dependabot**: Dependency updates for actions, npm, pip, docker
- **OSSF Scorecard**: Security posture assessment

### Step 4: SBOM & Signed Releases
- **SBOM Generation**: Creates CycloneDX software bill of materials
- **SBOM Monitoring**: Daily diff checking for new vulnerabilities
- **Artifact Signing**: Keyless signing with Cosign (for Docker images)

### Step 5: README Badges
Adds status badges after first heading:
```markdown
[![License](https://img.shields.io/github/license/owner/repo)](LICENSE)
[![CI](https://github.com/owner/repo/workflows/CI/badge.svg)](https://github.com/owner/repo/actions)
[![Security Rating](https://api.securityscorecards.dev/projects/github.com/owner/repo/badge)](https://api.securityscorecards.dev/projects/github.com/owner/repo)
[![SBOM](https://img.shields.io/badge/SBOM-Available-green)](docs/sbom/latest.json)
```

### Step 6: Stale Repository Archival
- Archives repositories named "Main-Project" with no commits >400 days
- Helps maintain clean repository lists

### Step 7: README Sections
Ensures these sections exist:
- `## ✨ Why this exists`
- `## ⚡ Quick Start`
- `## 🔍 Key Features`
- `## 🗺 Road Map`
- `## 🤝 Contributing`

### Step 8: Repository Pinning
- Pins top 6 repositories by star count
- Updates profile to showcase best work

### Step 9: Metrics Logging
Creates `metrics/profile_hygiene.json` with status tracking:
```json
{
  "repo_name": {
    "description_set": true,
    "topics_count": 8,
    "license_exists": true,
    "code_scanning": true,
    "dependabot": true,
    "scorecard": true,
    "sbom_workflow": true
  }
}
```

### Step 10: Pull Request Creation
- Creates branch: `repo-hygiene-bot-{timestamp}`
- Opens PR: `✨ repo-hygiene-bot update`
- Adds `automated-maintenance` label
- Assigns to repository owner

## 🛠 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | Yes | GitHub personal access token |
| `GITHUB_OWNER` | Yes | GitHub username or organization |

### Configuration File

Customize behavior in `hygiene_config.json`:

```json
{
  "metadata": {
    "default_description_template": "A project by {owner} - {repo_name}",
    "min_topics": 5,
    "max_topics": 8
  },
  "archival": {
    "stale_days": 400,
    "target_repo_name": "Main-Project"
  }
}
```

## 🔒 Security & Permissions

### Required GitHub Token Scopes
- `repo`: Full repository access
- `read:org`: Read organization membership
- `read:user`: Read user profile information

### Rate Limiting
- Automatically handles GitHub API rate limits
- Sleeps when rate limit exceeded
- Respects GitHub's secondary rate limits

### Security Best Practices
- Never logs or exposes GitHub tokens
- Uses authenticated API calls only
- Creates commits with bot identity
- Follows principle of least privilege

## 📊 Monitoring & Metrics

### Metrics Output
```json
{
  "repository_name": {
    "description_set": boolean,
    "topics_count": number,
    "license_exists": boolean,
    "code_scanning": boolean,
    "dependabot": boolean,
    "scorecard": boolean,
    "sbom_workflow": boolean
  }
}
```

### Change Tracking
All changes are logged and included in pull request descriptions:
- Repository metadata updates
- Community file creation
- Security workflow additions
- README modifications

## 🚀 Integration Options

### GitHub Actions
```yaml
name: Repository Hygiene
on:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  hygiene:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install requests
      - name: Run hygiene bot
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITHUB_OWNER: ${{ github.repository_owner }}
        run: python3 github_hygiene_bot.py --token "$GITHUB_TOKEN" --owner "$GITHUB_OWNER"
```

### Cron Job
```bash
# Add to crontab for weekly execution
0 0 * * 1 cd /path/to/hygiene-bot && ./run_hygiene_bot.sh
```

## 🔧 Troubleshooting

### Common Issues

**Permission Denied**
```bash
chmod +x run_hygiene_bot.sh
```

**Missing Dependencies**
```bash
pip install requests
```

**Rate Limit Exceeded**
- Bot automatically handles rate limits
- For large organizations, consider running during off-peak hours

**Token Permissions**
- Ensure token has `repo`, `read:org`, `read:user` scopes
- For organizations, may need additional permissions

### Debug Mode
```bash
# Add debug logging
export PYTHONPATH=.
python3 -u github_hygiene_bot.py --token "$GITHUB_TOKEN" --owner "$GITHUB_OWNER" 2>&1 | tee hygiene.log
```

## 📈 Benefits

### Security Improvements
- Automated vulnerability scanning
- Dependency update management
- Security policy standardization
- SBOM generation for supply chain security

### Community Health
- Standardized contribution guidelines
- Issue template consistency
- Code of conduct enforcement
- Clear project documentation

### Developer Experience
- Consistent repository structure
- Automated maintenance
- Clear project status via badges
- Improved discoverability via topics

### Compliance
- License compliance tracking
- Security scorecards
- Audit trail via pull requests
- Metrics for governance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/your-org/hygiene-bot
cd hygiene-bot
pip install -r requirements.txt
python -m pytest tests/
```

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## 🗺 Road Map

- [ ] Support for additional package ecosystems
- [ ] Custom template support
- [ ] Integration with external security tools
- [ ] Dashboard for hygiene metrics
- [ ] Slack/Teams notifications
- [ ] Policy-as-code configuration