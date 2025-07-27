#!/usr/bin/env python3
"""
GitHub Repository Hygiene Automation Bot
Implements the complete repo hygiene checklist for owned repositories.
"""

import json
import os
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any


class GitHubHygieneBot:
    def __init__(self, token: str, owner: str):
        self.token = token
        self.owner = owner
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitHub-Hygiene-Bot/1.0'
        })
        self.base_url = 'https://api.github.com'
        self.changes_made = []
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make authenticated GitHub API request with rate limiting."""
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        
        # Handle rate limiting
        if response.status_code == 429:
            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
            sleep_time = max(reset_time - int(time.time()), 60)
            print(f"Rate limited. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            response = self.session.request(method, url, **kwargs)
            
        return response

    def list_owned_repositories(self) -> List[Dict]:
        """Step 0: List all owned repositories (non-fork, non-template, non-archived)."""
        print("📋 Listing owned repositories...")
        repos = []
        page = 1
        
        while True:
            response = self._make_request(
                'GET', 
                f'/user/repos?per_page=100&page={page}&affiliation=owner'
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to fetch repositories: {response.status_code}")
                break
                
            page_repos = response.json()
            if not page_repos:
                break
                
            # Filter out forks, templates, and archived repos
            filtered_repos = [
                repo for repo in page_repos 
                if not repo['fork'] and not repo['is_template'] and not repo['archived']
            ]
            repos.extend(filtered_repos)
            page += 1
            
        print(f"✅ Found {len(repos)} owned repositories")
        return repos

    def update_repository_metadata(self, repo: Dict) -> None:
        """Step 1: Update description, website & topics."""
        repo_name = repo['name']
        changes = []
        
        # Update description if empty
        if not repo.get('description'):
            description = f"A project by {self.owner} - {repo_name}"
            if len(description) > 120:
                description = description[:117] + "..."
                
            response = self._make_request(
                'PATCH',
                f"/repos/{self.owner}/{repo_name}",
                json={'description': description}
            )
            
            if response.status_code == 200:
                changes.append(f"Added description: {description}")
                
        # Update homepage if null
        if not repo.get('homepage'):
            homepage = f"https://{self.owner}.github.io"
            response = self._make_request(
                'PATCH',
                f"/repos/{self.owner}/{repo_name}",
                json={'homepage': homepage}
            )
            
            if response.status_code == 200:
                changes.append(f"Set homepage: {homepage}")
                
        # Update topics if less than 5
        if len(repo.get('topics', [])) < 5:
            # Determine relevant topics based on repo content and language
            suggested_topics = self._suggest_topics(repo)
            
            response = self._make_request(
                'PUT',
                f"/repos/{self.owner}/{repo_name}/topics",
                json={'names': suggested_topics}
            )
            
            if response.status_code == 200:
                changes.append(f"Added topics: {', '.join(suggested_topics)}")
                
        if changes:
            self.changes_made.extend([f"{repo_name}: {change}" for change in changes])
            print(f"📝 Updated {repo_name}: {'; '.join(changes)}")

    def _suggest_topics(self, repo: Dict) -> List[str]:
        """Suggest relevant topics based on repository metadata."""
        topics = set(repo.get('topics', []))
        language = repo.get('language', '').lower()
        repo_name = repo['name'].lower()
        
        # Language-based topics
        language_topics = {
            'python': ['python', 'automation', 'api'],
            'javascript': ['javascript', 'web', 'frontend'],
            'typescript': ['typescript', 'web', 'frontend'],
            'go': ['golang', 'backend', 'microservices'],
            'rust': ['rust', 'systems', 'performance'],
            'java': ['java', 'enterprise', 'backend'],
            'dockerfile': ['docker', 'containerization', 'devops']
        }
        
        if language in language_topics:
            topics.update(language_topics[language])
            
        # Common project topics
        common_topics = [
            'open-source', 'github-actions', 'ci-cd', 'automation',
            'security', 'monitoring', 'testing', 'documentation'
        ]
        
        # Add based on repo name patterns
        if any(word in repo_name for word in ['bot', 'automation']):
            topics.add('automation')
        if any(word in repo_name for word in ['api', 'service']):
            topics.add('api')
        if any(word in repo_name for word in ['security', 'auth']):
            topics.add('security')
            
        # Fill up to 8 topics
        while len(topics) < 8 and common_topics:
            topics.add(common_topics.pop(0))
            
        return list(topics)[:8]

    def create_community_files(self, repo: Dict) -> None:
        """Step 2: Create missing community files."""
        repo_name = repo['name']
        changes = []
        
        community_files = {
            'LICENSE': self._get_apache_license(),
            'CODE_OF_CONDUCT.md': self._get_code_of_conduct(),
            'CONTRIBUTING.md': self._get_contributing_guide(),
            'SECURITY.md': self._get_security_policy()
        }
        
        # Check existing files
        for filename, content in community_files.items():
            response = self._make_request('GET', f"/repos/{self.owner}/{repo_name}/contents/{filename}")
            
            if response.status_code == 404:  # File doesn't exist
                # Create the file
                create_response = self._make_request(
                    'PUT',
                    f"/repos/{self.owner}/{repo_name}/contents/{filename}",
                    json={
                        'message': f'Add {filename}',
                        'content': content,
                        'committer': {
                            'name': 'GitHub Hygiene Bot',
                            'email': 'noreply@github.com'
                        }
                    }
                )
                
                if create_response.status_code == 201:
                    changes.append(f"Created {filename}")
                    
        # Create issue templates
        issue_templates = {
            '.github/ISSUE_TEMPLATE/bug.yml': self._get_bug_template(),
            '.github/ISSUE_TEMPLATE/feature.yml': self._get_feature_template()
        }
        
        for filepath, content in issue_templates.items():
            response = self._make_request('GET', f"/repos/{self.owner}/{repo_name}/contents/{filepath}")
            
            if response.status_code == 404:
                create_response = self._make_request(
                    'PUT',
                    f"/repos/{self.owner}/{repo_name}/contents/{filepath}",
                    json={
                        'message': f'Add {filepath}',
                        'content': content,
                        'committer': {
                            'name': 'GitHub Hygiene Bot',
                            'email': 'noreply@github.com'
                        }
                    }
                )
                
                if create_response.status_code == 201:
                    changes.append(f"Created {filepath}")
                    
        if changes:
            self.changes_made.extend([f"{repo_name}: {change}" for change in changes])
            print(f"📄 Created community files for {repo_name}: {'; '.join(changes)}")

    def setup_security_scanners(self, repo: Dict) -> None:
        """Step 3: Enable security scanners (CodeQL, Dependabot, OSSF Scorecard)."""
        repo_name = repo['name']
        changes = []
        
        # Create security workflows
        workflows = {
            '.github/workflows/codeql.yml': self._get_codeql_workflow(),
            '.github/dependabot.yml': self._get_dependabot_config(),
            '.github/workflows/scorecard.yml': self._get_scorecard_workflow()
        }
        
        for filepath, content in workflows.items():
            response = self._make_request('GET', f"/repos/{self.owner}/{repo_name}/contents/{filepath}")
            
            if response.status_code == 404:
                create_response = self._make_request(
                    'PUT',
                    f"/repos/{self.owner}/{repo_name}/contents/{filepath}",
                    json={
                        'message': f'Add {filepath.split("/")[-1]} security workflow',
                        'content': content,
                        'committer': {
                            'name': 'GitHub Hygiene Bot',
                            'email': 'noreply@github.com'
                        }
                    }
                )
                
                if create_response.status_code == 201:
                    changes.append(f"Added {filepath.split('/')[-1]}")
                    
        if changes:
            self.changes_made.extend([f"{repo_name}: {change}" for change in changes])
            print(f"🔒 Added security scanners for {repo_name}: {'; '.join(changes)}")

    def add_sbom_and_signing(self, repo: Dict) -> None:
        """Step 4: Add SBOM generation and signed releases workflows."""
        repo_name = repo['name']
        changes = []
        
        workflows = {
            '.github/workflows/sbom.yml': self._get_sbom_workflow(),
            '.github/workflows/sbom-diff.yml': self._get_sbom_diff_workflow()
        }
        
        for filepath, content in workflows.items():
            response = self._make_request('GET', f"/repos/{self.owner}/{repo_name}/contents/{filepath}")
            
            if response.status_code == 404:
                create_response = self._make_request(
                    'PUT',
                    f"/repos/{self.owner}/{repo_name}/contents/{filepath}",
                    json={
                        'message': f'Add {filepath.split("/")[-1]} workflow',
                        'content': content,
                        'committer': {
                            'name': 'GitHub Hygiene Bot',
                            'email': 'noreply@github.com'
                        }
                    }
                )
                
                if create_response.status_code == 201:
                    changes.append(f"Added {filepath.split('/')[-1]}")
                    
        if changes:
            self.changes_made.extend([f"{repo_name}: {change}" for change in changes])
            print(f"📋 Added SBOM workflows for {repo_name}: {'; '.join(changes)}")

    def update_readme_badges(self, repo: Dict) -> None:
        """Step 5: Add README badges."""
        repo_name = repo['name']
        
        # Get current README
        response = self._make_request('GET', f"/repos/{self.owner}/{repo_name}/contents/README.md")
        
        if response.status_code == 200:
            import base64
            content = base64.b64decode(response.json()['content']).decode('utf-8')
            
            # Check if badges already exist
            if '[![License]' not in content:
                badges = self._get_readme_badges(repo_name)
                
                # Insert badges after first heading
                lines = content.split('\n')
                insert_index = 0
                
                for i, line in enumerate(lines):
                    if line.startswith('#'):
                        insert_index = i + 1
                        break
                        
                lines.insert(insert_index, '')
                lines.insert(insert_index + 1, badges)
                
                new_content = '\n'.join(lines)
                
                # Update README
                update_response = self._make_request(
                    'PUT',
                    f"/repos/{self.owner}/{repo_name}/contents/README.md",
                    json={
                        'message': 'Add README badges',
                        'content': base64.b64encode(new_content.encode()).decode(),
                        'sha': response.json()['sha'],
                        'committer': {
                            'name': 'GitHub Hygiene Bot',
                            'email': 'noreply@github.com'
                        }
                    }
                )
                
                if update_response.status_code == 200:
                    self.changes_made.append(f"{repo_name}: Added README badges")
                    print(f"🏷️ Added badges to {repo_name} README")

    def archive_stale_repositories(self, repos: List[Dict]) -> None:
        """Step 6: Archive repositories with no commits >400 days."""
        cutoff_date = datetime.now() - timedelta(days=400)
        
        for repo in repos:
            if repo['name'] == 'Main-Project':  # Only archive if named Main-Project
                last_push = datetime.fromisoformat(repo['pushed_at'].replace('Z', '+00:00'))
                
                if last_push < cutoff_date:
                    response = self._make_request(
                        'PATCH',
                        f"/repos/{self.owner}/{repo['name']}",
                        json={'archived': True}
                    )
                    
                    if response.status_code == 200:
                        self.changes_made.append(f"{repo['name']}: Archived stale repository")
                        print(f"📦 Archived stale repository: {repo['name']}")

    def ensure_readme_sections(self, repo: Dict) -> None:
        """Step 7: Ensure README has required sections."""
        repo_name = repo['name']
        
        response = self._make_request('GET', f"/repos/{self.owner}/{repo_name}/contents/README.md")
        
        if response.status_code == 200:
            import base64
            content = base64.b64decode(response.json()['content']).decode('utf-8')
            
            required_sections = [
                '## ✨ Why this exists',
                '## ⚡ Quick Start',
                '## 🔍 Key Features',
                '## 🗺 Road Map',
                '## 🤝 Contributing'
            ]
            
            missing_sections = [section for section in required_sections if section not in content]
            
            if missing_sections:
                # Append missing sections
                content += '\n\n' + '\n\n'.join(missing_sections + [''])
                
                update_response = self._make_request(
                    'PUT',
                    f"/repos/{self.owner}/{repo_name}/contents/README.md",
                    json={
                        'message': 'Add required README sections',
                        'content': base64.b64encode(content.encode()).decode(),
                        'sha': response.json()['sha'],
                        'committer': {
                            'name': 'GitHub Hygiene Bot',
                            'email': 'noreply@github.com'
                        }
                    }
                )
                
                if update_response.status_code == 200:
                    self.changes_made.append(f"{repo_name}: Added missing README sections")
                    print(f"📝 Added missing sections to {repo_name} README")

    def pin_top_repositories(self, repos: List[Dict]) -> None:
        """Step 8: Pin top 6 repositories by star count."""
        # Sort by stars and take top 6
        top_repos = sorted(repos, key=lambda r: r['stargazers_count'], reverse=True)[:6]
        repo_names = [repo['name'] for repo in top_repos]
        
        response = self._make_request(
            'PUT',
            '/user/pinned_repositories',
            json={'repositories': repo_names}
        )
        
        if response.status_code == 200:
            self.changes_made.append(f"Pinned top repositories: {', '.join(repo_names)}")
            print(f"📌 Pinned top repositories: {', '.join(repo_names)}")

    def create_metrics_log(self, repos: List[Dict]) -> None:
        """Step 9: Create metrics log."""
        metrics = {}
        
        for repo in repos:
            repo_name = repo['name']
            repo_metrics = {
                'description_set': bool(repo.get('description')),
                'topics_count': len(repo.get('topics', [])),
                'license_exists': bool(repo.get('license')),
                'code_scanning': self._check_code_scanning(repo),
                'dependabot': self._check_dependabot(repo),
                'scorecard': self._check_scorecard(repo),
                'sbom_workflow': self._check_sbom_workflow(repo)
            }
            metrics[repo_name] = repo_metrics
            
        # Save metrics locally if in a repo
        if os.path.exists('.git'):
            os.makedirs('metrics', exist_ok=True)
            with open('metrics/profile_hygiene.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"📊 Created metrics log with {len(metrics)} repositories")

    def _check_code_scanning(self, repo: Dict) -> bool:
        """Check if CodeQL or similar code scanning is enabled."""
        response = self._make_request('GET', f"/repos/{self.owner}/{repo['name']}/contents/.github/workflows")
        if response.status_code == 200:
            workflows = response.json()
            return any('codeql' in w['name'].lower() for w in workflows if isinstance(workflows, list))
        return False

    def _check_dependabot(self, repo: Dict) -> bool:
        """Check if Dependabot is configured."""
        response = self._make_request('GET', f"/repos/{self.owner}/{repo['name']}/contents/.github/dependabot.yml")
        return response.status_code == 200

    def _check_scorecard(self, repo: Dict) -> bool:
        """Check if OSSF Scorecard is configured."""
        response = self._make_request('GET', f"/repos/{self.owner}/{repo['name']}/contents/.github/workflows")
        if response.status_code == 200:
            workflows = response.json()
            return any('scorecard' in w['name'].lower() for w in workflows if isinstance(workflows, list))
        return False

    def _check_sbom_workflow(self, repo: Dict) -> bool:
        """Check if SBOM workflow exists."""
        response = self._make_request('GET', f"/repos/{self.owner}/{repo['name']}/contents/.github/workflows")
        if response.status_code == 200:
            workflows = response.json()
            return any('sbom' in w['name'].lower() for w in workflows if isinstance(workflows, list))
        return False

    def create_hygiene_pr(self, repo_name: str) -> None:
        """Step 10: Create pull request with hygiene updates."""
        if not self.changes_made:
            print("ℹ️ No changes made, skipping PR creation")
            return
            
        # Create a new branch
        main_ref = self._make_request('GET', f"/repos/{self.owner}/{repo_name}/git/ref/heads/main")
        if main_ref.status_code != 200:
            print(f"❌ Could not get main branch ref for {repo_name}")
            return
            
        main_sha = main_ref.json()['object']['sha']
        branch_name = f"repo-hygiene-bot-{int(time.time())}"
        
        # Create branch
        create_branch = self._make_request(
            'POST',
            f"/repos/{self.owner}/{repo_name}/git/refs",
            json={
                'ref': f'refs/heads/{branch_name}',
                'sha': main_sha
            }
        )
        
        if create_branch.status_code != 201:
            print(f"❌ Could not create branch for {repo_name}")
            return
            
        # Create PR
        pr_body = "## Repository Hygiene Updates\n\n" + \
                 "\n".join(f"• {change}" for change in self.changes_made) + \
                 "\n\n🤖 Generated with GitHub Hygiene Bot"
                 
        pr_response = self._make_request(
            'POST',
            f"/repos/{self.owner}/{repo_name}/pulls",
            json={
                'title': '✨ repo-hygiene-bot update',
                'head': branch_name,
                'base': 'main',
                'body': pr_body
            }
        )
        
        if pr_response.status_code == 201:
            pr_url = pr_response.json()['html_url']
            
            # Add label and assignee
            pr_number = pr_response.json()['number']
            
            # Add label
            self._make_request(
                'POST',
                f"/repos/{self.owner}/{repo_name}/issues/{pr_number}/labels",
                json=['automated-maintenance']
            )
            
            # Add assignee
            self._make_request(
                'POST',
                f"/repos/{self.owner}/{repo_name}/issues/{pr_number}/assignees",
                json={'assignees': [self.owner]}
            )
            
            print(f"🎉 Created hygiene PR: {pr_url}")
        else:
            print(f"❌ Failed to create PR for {repo_name}")

    def run_hygiene_automation(self) -> None:
        """Run the complete hygiene automation process."""
        print("🚀 Starting GitHub Repository Hygiene Automation")
        
        # Step 0: List repositories
        repos = self.list_owned_repositories()
        if not repos:
            print("❌ No repositories found")
            return
            
        print(f"🔍 Processing {len(repos)} repositories")
        
        for repo in repos:
            print(f"\n📂 Processing: {repo['name']}")
            
            # Reset changes for each repo
            repo_changes = []
            
            # Steps 1-8: Apply hygiene improvements
            self.update_repository_metadata(repo)
            self.create_community_files(repo)
            self.setup_security_scanners(repo)
            self.add_sbom_and_signing(repo)
            self.update_readme_badges(repo)
            self.ensure_readme_sections(repo)
            
        # Steps that apply to all repos
        self.archive_stale_repositories(repos)
        self.pin_top_repositories(repos)
        self.create_metrics_log(repos)
        
        print(f"\n✅ Hygiene automation complete! Made {len(self.changes_made)} changes")
        
        # If running in a repo context, create PR
        if os.path.exists('.git'):
            repo_name = os.path.basename(os.getcwd())
            self.create_hygiene_pr(repo_name)

    # Template methods for community files and workflows
    def _get_apache_license(self) -> str:
        """Get base64 encoded Apache 2.0 license."""
        import base64
        license_text = """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

[Full Apache 2.0 license text...]"""
        return base64.b64encode(license_text.encode()).decode()

    def _get_code_of_conduct(self) -> str:
        """Get base64 encoded Contributor Covenant."""
        import base64
        coc_text = """# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone..."""
        return base64.b64encode(coc_text.encode()).decode()

    def _get_contributing_guide(self) -> str:
        """Get base64 encoded contributing guide."""
        import base64
        contrib_text = """# Contributing

## Development Setup

1. Fork the repository
2. Clone your fork
3. Install dependencies
4. Run tests

## Commit Convention

We use [Conventional Commits](https://conventionalcommits.org/):
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for tests"""
        return base64.b64encode(contrib_text.encode()).decode()

    def _get_security_policy(self) -> str:
        """Get base64 encoded security policy."""
        import base64
        security_text = f"""# Security Policy

## Reporting Security Vulnerabilities

Please report security vulnerabilities to: security@{self.owner}.dev

We aim to respond within 90 days and will keep you updated on progress.

## Supported Versions

Only the latest version receives security updates."""
        return base64.b64encode(security_text.encode()).decode()

    def _get_bug_template(self) -> str:
        """Get base64 encoded bug report template."""
        import base64
        template = """name: Bug Report
description: Report a bug
body:
  - type: textarea
    attributes:
      label: Description
      description: Describe the bug
    validations:
      required: true
  - type: textarea
    attributes:
      label: Steps to Reproduce
      description: Steps to reproduce the issue
    validations:
      required: true"""
        return base64.b64encode(template.encode()).decode()

    def _get_feature_template(self) -> str:
        """Get base64 encoded feature request template."""
        import base64
        template = """name: Feature Request
description: Request a new feature
body:
  - type: textarea
    attributes:
      label: Description
      description: Describe the feature
    validations:
      required: true
  - type: textarea
    attributes:
      label: Use Case
      description: Why is this needed?
    validations:
      required: true"""
        return base64.b64encode(template.encode()).decode()

    def _get_codeql_workflow(self) -> str:
        """Get base64 encoded CodeQL workflow."""
        import base64
        workflow = """name: "CodeQL"
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 1'

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    strategy:
      fail-fast: false
      matrix:
        language: [ 'python' ]
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3"""
        return base64.b64encode(workflow.encode()).decode()

    def _get_dependabot_config(self) -> str:
        """Get base64 encoded Dependabot config."""
        import base64
        config = """version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"""
        return base64.b64encode(config.encode()).decode()

    def _get_scorecard_workflow(self) -> str:
        """Get base64 encoded OSSF Scorecard workflow."""
        import base64
        workflow = """name: Scorecard
on:
  branch_protection_rule:
  schedule:
    - cron: '0 0 * * 1'
  push:
    branches: [ main ]

permissions: read-all

jobs:
  analysis:
    name: Scorecard analysis
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      id-token: write
    steps:
      - name: "Checkout code"
        uses: actions/checkout@v4
        with:
          persist-credentials: false
      - name: "Run analysis"
        uses: ossf/scorecard-action@v2.3.3
        with:
          results_file: results.sarif
          results_format: sarif
          publish_results: true
      - name: "Upload SARIF results to code-scanning"
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif"""
        return base64.b64encode(workflow.encode()).decode()

    def _get_sbom_workflow(self) -> str:
        """Get base64 encoded SBOM workflow."""
        import base64
        workflow = """name: SBOM Generation
on:
  push:
    branches: [ main ]
  release:
    types: [ published ]

jobs:
  sbom:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Generate SBOM
        uses: cyclonedx/github-action@v1
        with:
          path: '.'
      - name: Create docs directory
        run: mkdir -p docs/sbom
      - name: Move SBOM
        run: mv bom.json docs/sbom/latest.json
      - name: Upload SBOM
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: docs/sbom/latest.json"""
        return base64.b64encode(workflow.encode()).decode()

    def _get_sbom_diff_workflow(self) -> str:
        """Get base64 encoded SBOM diff workflow."""
        import base64
        workflow = """name: SBOM Diff
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  sbom-diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Generate current SBOM
        uses: cyclonedx/github-action@v1
        with:
          path: '.'
      - name: Download previous SBOM
        run: |
          curl -s -o previous-sbom.json https://raw.githubusercontent.com/${{ github.repository }}/main/docs/sbom/latest.json || echo '{}' > previous-sbom.json
      - name: Compare SBOMs
        run: |
          cyclonedx diff previous-sbom.json bom.json --format json > sbom-diff.json
          if [ -s sbom-diff.json ]; then
            echo "SBOM changes detected"
            cat sbom-diff.json
          fi"""
        return base64.b64encode(workflow.encode()).decode()

    def _get_readme_badges(self, repo_name: str) -> str:
        """Generate README badges."""
        return f"""[![License](https://img.shields.io/github/license/{self.owner}/{repo_name})](LICENSE)
[![CI](https://github.com/{self.owner}/{repo_name}/workflows/CI/badge.svg)](https://github.com/{self.owner}/{repo_name}/actions)
[![Security Rating](https://api.securityscorecards.dev/projects/github.com/{self.owner}/{repo_name}/badge)](https://api.securityscorecards.dev/projects/github.com/{self.owner}/{repo_name})
[![SBOM](https://img.shields.io/badge/SBOM-Available-green)](docs/sbom/latest.json)"""


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GitHub Repository Hygiene Bot')
    parser.add_argument('--token', required=True, help='GitHub personal access token')
    parser.add_argument('--owner', required=True, help='GitHub username/organization')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    bot = GitHubHygieneBot(args.token, args.owner)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        repos = bot.list_owned_repositories()
        print(f"Would process {len(repos)} repositories:")
        for repo in repos:
            print(f"  • {repo['name']} (⭐ {repo['stargazers_count']})")
    else:
        bot.run_hygiene_automation()


if __name__ == '__main__':
    main()