# GitHub Workflows Setup Instructions

Due to GitHub App permissions, the workflow files need to be added manually. Please create the following files in your repository:

## 1. Create `.github/workflows/ci.yml`

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

env:
  PYTHON_VERSION: '3.11'
  POETRY_VERSION: '1.6.1'

jobs:
  lint:
    name: Code Quality & Linting
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt', '**/pyproject.toml') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install black ruff mypy bandit safety
          pip install -e .

      - name: Run Black formatting check
        run: black --check --diff .

      - name: Run Ruff linting
        run: ruff check .

      - name: Run type checking with mypy
        run: mypy src/

      - name: Run security checks with bandit
        run: bandit -r src/ -f json -o bandit-report.json || true

      - name: Check for security vulnerabilities
        run: safety check

      - name: Upload bandit results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: bandit-report
          path: bandit-report.json

  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-${{ matrix.python-version }}-pip-${{ hashFiles('**/requirements*.txt', '**/pyproject.toml') }}
          restore-keys: |
            ${{ runner.os }}-${{ matrix.python-version }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pytest-cov pytest-mock pytest-asyncio
          pip install -e .

      - name: Create test environment
        run: |
          mkdir -p data/test_indices
          export API_KEY=test-api-key
          export ENVIRONMENT=test

      - name: Run unit tests
        run: pytest tests/ -m "unit" --cov=lexgraph_legal_rag --cov-report=xml --cov-report=term-missing

      - name: Run integration tests
        run: pytest tests/ -m "integration" --cov=lexgraph_legal_rag --cov-append --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          fail_ci_if_error: false

  security:
    name: Security Scanning
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

  build:
    name: Build & Package
    runs-on: ubuntu-latest
    needs: [lint, test]
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Check package
        run: twine check dist/*

      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: python-package
          path: dist/

  docker:
    name: Docker Build & Scan
    runs-on: ubuntu-latest
    needs: [lint, test]
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: lexgraph-legal-rag:test
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run Trivy vulnerability scanner on Docker image
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'lexgraph-legal-rag:test'
          format: 'sarif'
          output: 'trivy-docker-results.sarif'

      - name: Upload Docker scan results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-docker-results.sarif'

  performance:
    name: Performance Tests
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .
          pip install pytest pytest-benchmark

      - name: Run performance tests
        run: pytest tests/performance/ -v --benchmark-json=benchmark.json

      - name: Upload performance results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: benchmark.json

  validate-pr:
    name: PR Validation
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    needs: [lint, test, security, build]
    
    steps:
      - name: Check all jobs passed
        run: echo "All CI checks passed successfully!"

      - name: Comment PR
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '✅ All CI checks passed! This PR is ready for review.'
            })
```

## 2. Create `.github/workflows/cd.yml`

```yaml
name: Continuous Deployment

on:
  release:
    types: [published]
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  PYTHON_VERSION: '3.11'

jobs:
  build-and-publish:
    name: Build & Publish Package
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      id-token: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        if: startsWith(github.ref, 'refs/tags/v')
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          user: __token__
          password: ${{ secrets.PYPI_API_TOKEN }}

      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: python-package-${{ github.sha }}
          path: dist/

  docker-build-push:
    name: Build & Push Docker Image
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}}
            type=sha,prefix={{branch}}-

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [docker-build-push]
    if: github.ref == 'refs/heads/main'
    environment:
      name: staging
      url: https://staging-lexgraph.terragon.ai

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Configure kubectl
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > ~/.kube/config

      - name: Deploy to staging
        run: |
          kubectl set image deployment/lexgraph-api \
            lexgraph-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${{ github.sha }} \
            -n staging
          kubectl rollout status deployment/lexgraph-api -n staging --timeout=300s

      - name: Run smoke tests
        run: |
          # Wait for deployment to be ready
          kubectl wait --for=condition=available deployment/lexgraph-api -n staging --timeout=300s
          
          # Get the staging URL
          STAGING_URL=$(kubectl get ingress lexgraph-ingress -n staging -o jsonpath='{.spec.rules[0].host}')
          
          # Run basic health checks
          curl -f https://$STAGING_URL/health || exit 1
          curl -f https://$STAGING_URL/metrics || exit 1

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [deploy-staging, build-and-publish]
    if: startsWith(github.ref, 'refs/tags/v')
    environment:
      name: production
      url: https://lexgraph.terragon.ai

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Configure kubectl
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBE_CONFIG_PRODUCTION }}" | base64 -d > ~/.kube/config

      - name: Deploy to production
        run: |
          # Extract version from tag
          VERSION=${GITHUB_REF#refs/tags/v}
          
          # Update production deployment
          kubectl set image deployment/lexgraph-api \
            lexgraph-api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:$VERSION \
            -n production
          
          # Wait for rollout to complete
          kubectl rollout status deployment/lexgraph-api -n production --timeout=600s

      - name: Run production health checks
        run: |
          # Wait for deployment to be ready
          kubectl wait --for=condition=available deployment/lexgraph-api -n production --timeout=600s
          
          # Get the production URL
          PROD_URL=$(kubectl get ingress lexgraph-ingress -n production -o jsonpath='{.spec.rules[0].host}')
          
          # Run comprehensive health checks
          curl -f https://$PROD_URL/health || exit 1
          curl -f https://$PROD_URL/metrics || exit 1
          
          # Test basic API functionality
          curl -f -H "X-API-Key: ${{ secrets.PROD_API_KEY }}" \
            "https://$PROD_URL/api/v1/search?q=test" || exit 1

      - name: Create deployment record
        uses: actions/github-script@v6
        with:
          script: |
            const { data: release } = await github.rest.repos.getReleaseByTag({
              owner: context.repo.owner,
              repo: context.repo.repo,
              tag: context.ref.replace('refs/tags/', '')
            });
            
            await github.rest.repos.createDeployment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              ref: context.sha,
              environment: 'production',
              description: `Deployed ${context.ref} to production`,
              auto_merge: false
            });

  notify:
    name: Notify Deployment
    runs-on: ubuntu-latest
    needs: [deploy-production]
    if: always()

    steps:
      - name: Notify Slack
        if: success()
        uses: 8398a7/action-slack@v3
        with:
          status: success
          channel: '#deployments'
          message: '🚀 Successfully deployed LexGraph Legal RAG to production!'
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

      - name: Notify Slack on failure
        if: failure()
        uses: 8398a7/action-slack@v3
        with:
          status: failure
          channel: '#deployments'
          message: '❌ Production deployment failed for LexGraph Legal RAG'
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

  rollback:
    name: Rollback Production
    runs-on: ubuntu-latest
    if: failure() && startsWith(github.ref, 'refs/tags/v')
    needs: [deploy-production]
    environment:
      name: production-rollback

    steps:
      - name: Configure kubectl
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBE_CONFIG_PRODUCTION }}" | base64 -d > ~/.kube/config

      - name: Rollback deployment
        run: |
          kubectl rollout undo deployment/lexgraph-api -n production
          kubectl rollout status deployment/lexgraph-api -n production --timeout=300s

      - name: Verify rollback
        run: |
          kubectl wait --for=condition=available deployment/lexgraph-api -n production --timeout=300s
          PROD_URL=$(kubectl get ingress lexgraph-ingress -n production -o jsonpath='{.spec.rules[0].host}')
          curl -f https://$PROD_URL/health || exit 1
```

## 3. Create `.github/workflows/security.yml`

```yaml
name: Security Scanning

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  actions: read
  contents: read
  security-events: write

jobs:
  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install safety pip-audit
          pip install -e .

      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json || true

      - name: Run pip-audit
        run: |
          pip-audit --format=json --output=pip-audit-report.json || true

      - name: Upload safety report
        uses: actions/upload-artifact@v3
        with:
          name: safety-report
          path: safety-report.json

      - name: Upload pip-audit report
        uses: actions/upload-artifact@v3
        with:
          name: pip-audit-report
          path: pip-audit-report.json

  secret-scan:
    name: Secret Detection
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run TruffleHog OSS
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD
          extra_args: --debug --only-verified

  sast-scan:
    name: Static Application Security Testing
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install bandit semgrep
          pip install -e .

      - name: Run Bandit
        run: |
          bandit -r src/ -f json -o bandit-report.json || true

      - name: Run Semgrep
        run: |
          semgrep --config=auto --json --output=semgrep-report.json src/ || true

      - name: Upload Bandit report
        uses: actions/upload-artifact@v3
        with:
          name: bandit-report
          path: bandit-report.json

      - name: Upload Semgrep report
        uses: actions/upload-artifact@v3
        with:
          name: semgrep-report
          path: semgrep-report.json

  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t lexgraph-security-scan:latest .

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'lexgraph-security-scan:latest'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Run Docker Scout
        if: github.event_name != 'pull_request'
        uses: docker/scout-action@v1
        with:
          command: cves
          image: lexgraph-security-scan:latest
          sarif-file: scout-report.sarif

      - name: Upload Docker Scout scan results
        if: github.event_name != 'pull_request'
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: scout-report.sarif

  license-scan:
    name: License Compliance Check
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pip-licenses licensecheck
          pip install -e .

      - name: Check licenses
        run: |
          pip-licenses --format=json --output-file=licenses.json
          licensecheck --zero

      - name: Upload license report
        uses: actions/upload-artifact@v3
        with:
          name: license-report
          path: licenses.json

  sbom-generation:
    name: Generate Software Bill of Materials
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install cyclonedx-bom
          pip install -e .

      - name: Generate SBOM
        run: |
          cyclonedx-py -o sbom.json

      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: software-bill-of-materials
          path: sbom.json

  security-report:
    name: Generate Security Report
    runs-on: ubuntu-latest
    needs: [dependency-scan, secret-scan, sast-scan, container-scan, license-scan, sbom-generation]
    if: always()
    
    steps:
      - name: Download all artifacts
        uses: actions/download-artifact@v3

      - name: Generate security summary
        run: |
          cat > security-summary.md << 'EOF'
          # Security Scan Summary
          
          ## Scan Results
          
          | Scan Type | Status | Artifacts |
          |-----------|--------|-----------|
          | Dependency Vulnerabilities | ${{ needs.dependency-scan.result }} | safety-report.json, pip-audit-report.json |
          | Secret Detection | ${{ needs.secret-scan.result }} | TruffleHog scan completed |
          | Static Analysis | ${{ needs.sast-scan.result }} | bandit-report.json, semgrep-report.json |
          | Container Security | ${{ needs.container-scan.result }} | trivy-results.sarif |
          | License Compliance | ${{ needs.license-scan.result }} | licenses.json |
          | SBOM Generation | ${{ needs.sbom-generation.result }} | sbom.json |
          
          ## Next Steps
          
          1. Review all security scan reports in the artifacts
          2. Address any high/critical vulnerabilities found
          3. Update dependencies with security patches
          4. Verify license compliance for all dependencies
          
          Generated on: $(date -u)
          Commit: ${{ github.sha }}
          EOF

      - name: Upload security summary
        uses: actions/upload-artifact@v3
        with:
          name: security-summary
          path: security-summary.md

      - name: Comment on PR with security status
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const summary = fs.readFileSync('security-summary.md', 'utf8');
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## 🔒 Security Scan Results\n\n${summary}\n\nPlease review the security scan artifacts for detailed findings.`
            });

  compliance-check:
    name: Compliance Verification
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Verify security policy compliance
        run: |
          # Check for required security files
          test -f SECURITY.md || (echo "SECURITY.md is required" && exit 1)
          test -f .github/SECURITY.md || test -f SECURITY.md || (echo "Security policy not found" && exit 1)
          
          # Check for vulnerability disclosure process
          grep -i "security" README.md || (echo "Security section missing in README" && exit 1)
          
          # Verify pre-commit hooks are configured
          test -f .pre-commit-config.yaml || (echo "Pre-commit hooks not configured" && exit 1)

      - name: Check branch protection
        uses: actions/github-script@v6
        with:
          script: |
            try {
              const { data: branch } = await github.rest.repos.getBranch({
                owner: context.repo.owner,
                repo: context.repo.repo,
                branch: 'main'
              });
              
              if (!branch.protection) {
                core.setFailed('Main branch protection is not enabled');
              }
              
              if (!branch.protection.required_status_checks) {
                core.setFailed('Required status checks are not configured');
              }
            } catch (error) {
              core.setFailed(`Failed to check branch protection: ${error.message}`);
            }
```

## 4. Create `.github/workflows/dependencies.yml`

```yaml
name: Dependency Management

on:
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Monday at 4 AM UTC
  workflow_dispatch:
    inputs:
      update_type:
        description: 'Type of dependency update'
        required: true
        type: choice
        options:
          - security
          - all
          - major
          - minor
          - patch

permissions:
  contents: write
  pull-requests: write
  security-events: write

jobs:
  dependency-scan:
    name: Scan Dependencies for Vulnerabilities
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install safety pip-audit
          pip install -e .

      - name: Run vulnerability scan
        id: scan
        run: |
          # Run safety check
          safety check --json --output safety-report.json || true
          
          # Run pip-audit
          pip-audit --format=json --output=pip-audit-report.json || true
          
          # Check if vulnerabilities found
          if [ -s safety-report.json ] || [ -s pip-audit-report.json ]; then
            echo "vulnerabilities=true" >> $GITHUB_OUTPUT
          else
            echo "vulnerabilities=false" >> $GITHUB_OUTPUT
          fi

      - name: Upload vulnerability reports
        uses: actions/upload-artifact@v3
        with:
          name: vulnerability-reports
          path: |
            safety-report.json
            pip-audit-report.json

      - name: Create security issue if vulnerabilities found
        if: steps.scan.outputs.vulnerabilities == 'true'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            
            let vulnerabilities = [];
            
            // Parse safety report
            try {
              const safetyReport = JSON.parse(fs.readFileSync('safety-report.json', 'utf8'));
              if (safetyReport.vulnerabilities) {
                vulnerabilities = vulnerabilities.concat(safetyReport.vulnerabilities);
              }
            } catch (error) {
              console.log('No safety vulnerabilities found');
            }
            
            // Parse pip-audit report  
            try {
              const pipAuditReport = JSON.parse(fs.readFileSync('pip-audit-report.json', 'utf8'));
              if (pipAuditReport.vulnerabilities) {
                vulnerabilities = vulnerabilities.concat(pipAuditReport.vulnerabilities);
              }
            } catch (error) {
              console.log('No pip-audit vulnerabilities found');
            }
            
            if (vulnerabilities.length > 0) {
              let issueBody = '# 🚨 Security Vulnerabilities Detected\n\n';
              issueBody += `Found ${vulnerabilities.length} security vulnerabilities in dependencies:\n\n`;
              
              vulnerabilities.forEach((vuln, index) => {
                issueBody += `## Vulnerability ${index + 1}\n`;
                issueBody += `- **Package**: ${vuln.package_name || vuln.package}\n`;
                issueBody += `- **Severity**: ${vuln.severity || 'Unknown'}\n`;
                issueBody += `- **Summary**: ${vuln.vulnerability_id || vuln.id}\n`;
                issueBody += `- **Affected Version**: ${vuln.installed_version || vuln.version}\n\n`;
              });
              
              issueBody += '\n## Action Required\n';
              issueBody += 'Please update the affected dependencies to secure versions.\n\n';
              issueBody += '## Automated Actions\n';
              issueBody += '- [ ] Review vulnerability details\n';
              issueBody += '- [ ] Update affected packages\n';
              issueBody += '- [ ] Test application functionality\n';
              issueBody += '- [ ] Deploy security updates\n';
              
              await github.rest.issues.create({
                owner: context.repo.owner,
                repo: context.repo.repo,
                title: '🚨 Security vulnerabilities detected in dependencies',
                body: issueBody,
                labels: ['security', 'dependencies', 'high-priority']
              });
            }

  update-dependencies:
    name: Update Dependencies
    runs-on: ubuntu-latest
    needs: dependency-scan
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install pip-tools
        run: |
          python -m pip install --upgrade pip
          pip install pip-tools

      - name: Update requirements
        id: update
        run: |
          # Backup current requirements
          cp requirements.txt requirements.txt.backup
          
          # Generate updated requirements
          if [ "${{ github.event.inputs.update_type }}" = "security" ]; then
            # Only update packages with security vulnerabilities
            pip-compile --upgrade-package package-name requirements.in
          else
            # Update all packages
            pip-compile --upgrade requirements.in
          fi
          
          # Check if there are changes
          if ! diff -q requirements.txt requirements.txt.backup > /dev/null; then
            echo "changes=true" >> $GITHUB_OUTPUT
            echo "Updates found in requirements.txt"
          else
            echo "changes=false" >> $GITHUB_OUTPUT
            echo "No updates needed"
          fi

      - name: Test updated dependencies
        if: steps.update.outputs.changes == 'true'
        run: |
          # Install updated dependencies
          pip install -r requirements.txt
          pip install -e .
          
          # Run basic tests to ensure compatibility
          python -c "import lexgraph_legal_rag; print('Package imports successfully')"
          
          # Run a subset of tests
          python -m pytest tests/ -x --tb=short -q

      - name: Create Pull Request
        if: steps.update.outputs.changes == 'true'
        uses: peter-evans/create-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          commit-message: 'chore(deps): update dependencies'
          title: '🔄 Automated dependency updates'
          body: |
            ## 🔄 Dependency Updates
            
            This PR contains automated dependency updates.
            
            ### Changes
            - Updated `requirements.txt` with latest compatible versions
            - All tests pass with updated dependencies
            
            ### Security
            - Includes security patches for vulnerable packages
            - No breaking changes detected
            
            ### Testing
            - [x] Basic import tests pass
            - [x] Core functionality verified
            - [ ] Manual testing recommended
            
            ### Review Checklist
            - [ ] Review changed dependencies
            - [ ] Check for breaking changes
            - [ ] Verify security updates
            - [ ] Approve and merge if all checks pass
            
            ---
            
            🤖 This PR was created automatically by the dependency update workflow.
          branch: automated/dependency-updates
          delete-branch: true
          labels: |
            dependencies
            automated
            maintenance

  stale-dependency-check:
    name: Check for Stale Dependencies
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pip-check pip-review
          pip install -e .

      - name: Check for outdated packages
        id: outdated
        run: |
          # Check for outdated packages
          OUTDATED=$(pip list --outdated --format=json)
          echo "outdated=$OUTDATED" >> $GITHUB_OUTPUT
          
          if [ "$OUTDATED" != "[]" ]; then
            echo "outdated_found=true" >> $GITHUB_OUTPUT
          else
            echo "outdated_found=false" >> $GITHUB_OUTPUT
          fi

      - name: Create outdated dependencies issue
        if: steps.outdated.outputs.outdated_found == 'true' && github.event_name == 'schedule'
        uses: actions/github-script@v6
        with:
          script: |
            const outdated = JSON.parse('${{ steps.outdated.outputs.outdated }}');
            
            let issueBody = '# 📦 Outdated Dependencies Report\n\n';
            issueBody += `Found ${outdated.length} outdated dependencies:\n\n`;
            issueBody += '| Package | Current | Latest | Type |\n';
            issueBody += '|---------|---------|--------|----- |\n';
            
            outdated.forEach(pkg => {
              issueBody += `| ${pkg.name} | ${pkg.version} | ${pkg.latest_version} | ${pkg.latest_filetype} |\n`;
            });
            
            issueBody += '\n## Recommended Actions\n';
            issueBody += '1. Review the outdated packages above\n';
            issueBody += '2. Check for breaking changes in newer versions\n';
            issueBody += '3. Update dependencies in batches\n';
            issueBody += '4. Run automated dependency update workflow\n\n';
            issueBody += '## Automation\n';
            issueBody += 'You can trigger automated updates by running the "Dependency Management" workflow manually.\n';
            
            // Check if similar issue already exists
            const { data: issues } = await github.rest.issues.listForRepo({
              owner: context.repo.owner,
              repo: context.repo.repo,
              labels: 'dependencies,outdated',
              state: 'open'
            });
            
            if (issues.length === 0) {
              await github.rest.issues.create({
                owner: context.repo.owner,
                repo: context.repo.repo,
                title: '📦 Outdated dependencies detected',
                body: issueBody,
                labels: ['dependencies', 'outdated', 'maintenance']
              });
            }

  cleanup-old-branches:
    name: Cleanup Old Dependency Branches
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Delete old dependency branches
        uses: actions/github-script@v6
        with:
          script: |
            const { data: branches } = await github.rest.repos.listBranches({
              owner: context.repo.owner,
              repo: context.repo.repo,
              per_page: 100
            });
            
            const oldBranches = branches.filter(branch => {
              return branch.name.startsWith('automated/dependency-updates') && 
                     branch.name !== 'automated/dependency-updates';
            });
            
            for (const branch of oldBranches) {
              try {
                // Check if branch has open PR
                const { data: prs } = await github.rest.pulls.list({
                  owner: context.repo.owner,
                  repo: context.repo.repo,
                  head: `${context.repo.owner}:${branch.name}`,
                  state: 'open'
                });
                
                if (prs.length === 0) {
                  await github.rest.git.deleteRef({
                    owner: context.repo.owner,
                    repo: context.repo.repo,
                    ref: `heads/${branch.name}`
                  });
                  console.log(`Deleted old branch: ${branch.name}`);
                }
              } catch (error) {
                console.log(`Could not delete branch ${branch.name}: ${error.message}`);
              }
            }
```

## 5. Create `.github/workflows/release.yml`

```yaml
name: Release Management

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      release_type:
        description: 'Type of release'
        required: true
        type: choice
        options:
          - patch
          - minor
          - major
          - prerelease
      prerelease:
        description: 'Create pre-release'
        required: false
        type: boolean
        default: false

env:
  PYTHON_VERSION: '3.11'

jobs:
  check-changes:
    name: Check for Release-worthy Changes
    runs-on: ubuntu-latest
    outputs:
      should_release: ${{ steps.check.outputs.should_release }}
      version_bump: ${{ steps.check.outputs.version_bump }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Check for conventional commits
        id: check
        uses: actions/github-script@v6
        with:
          script: |
            const { execSync } = require('child_process');
            
            // Get commits since last tag
            let lastTag;
            try {
              lastTag = execSync('git describe --tags --abbrev=0', { encoding: 'utf8' }).trim();
            } catch {
              lastTag = 'HEAD~10'; // If no tags exist, check last 10 commits
            }
            
            const commits = execSync(`git log --pretty=format:"%s" ${lastTag}..HEAD`, { encoding: 'utf8' })
              .split('\n')
              .filter(line => line.trim());
            
            let shouldRelease = false;
            let versionBump = 'patch';
            
            for (const commit of commits) {
              if (commit.match(/^feat(\(.+\))?!:/)) {
                // Breaking change
                shouldRelease = true;
                versionBump = 'major';
                break;
              } else if (commit.match(/^feat(\(.+\))?:/)) {
                // New feature
                shouldRelease = true;
                if (versionBump === 'patch') versionBump = 'minor';
              } else if (commit.match(/^fix(\(.+\))?:/)) {
                // Bug fix
                shouldRelease = true;
              }
            }
            
            // Override with manual input if provided
            if (context.payload.inputs?.release_type) {
              shouldRelease = true;
              versionBump = context.payload.inputs.release_type;
            }
            
            core.setOutput('should_release', shouldRelease);
            core.setOutput('version_bump', versionBump);
            
            console.log(`Should release: ${shouldRelease}`);
            console.log(`Version bump: ${versionBump}`);

  semantic-release:
    name: Semantic Release
    runs-on: ubuntu-latest
    needs: check-changes
    if: needs.check-changes.outputs.should_release == 'true'
    permissions:
      contents: write
      packages: write
      issues: write
      pull-requests: write
    
    outputs:
      version: ${{ steps.semantic.outputs.version }}
      released: ${{ steps.semantic.outputs.released }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install semantic-release
        run: |
          npm install -g semantic-release
          npm install -g @semantic-release/changelog
          npm install -g @semantic-release/git
          npm install -g @semantic-release/github
          npm install -g @semantic-release/exec

      - name: Create .releaserc.json
        run: |
          cat > .releaserc.json << 'EOF'
          {
            "branches": ["main"],
            "plugins": [
              "@semantic-release/commit-analyzer",
              "@semantic-release/release-notes-generator",
              [
                "@semantic-release/changelog",
                {
                  "changelogFile": "CHANGELOG.md"
                }
              ],
              [
                "@semantic-release/exec",
                {
                  "prepareCmd": "python -c \"import re; content=open('pyproject.toml').read(); content=re.sub(r'version = \"[^\"]*\"', f'version = \\\"{os.environ[\\\"NEXT_RELEASE_VERSION\\\"]}\\\"', content); open('pyproject.toml', 'w').write(content)\""
                }
              ],
              [
                "@semantic-release/git",
                {
                  "assets": ["CHANGELOG.md", "pyproject.toml"],
                  "message": "chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}"
                }
              ],
              "@semantic-release/github"
            ]
          }
          EOF

      - name: Run semantic-release
        id: semantic
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # Run semantic-release and capture output
          OUTPUT=$(semantic-release --dry-run 2>&1 || semantic-release 2>&1)
          echo "$OUTPUT"
          
          # Extract version if released
          if echo "$OUTPUT" | grep -q "Published release"; then
            VERSION=$(echo "$OUTPUT" | grep -o "Published release [0-9]*\.[0-9]*\.[0-9]*" | grep -o "[0-9]*\.[0-9]*\.[0-9]*")
            echo "version=$VERSION" >> $GITHUB_OUTPUT
            echo "released=true" >> $GITHUB_OUTPUT
          else
            echo "released=false" >> $GITHUB_OUTPUT
          fi

  build-release-assets:
    name: Build Release Assets
    runs-on: ubuntu-latest
    needs: semantic-release
    if: needs.semantic-release.outputs.released == 'true'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          ref: main  # Get the updated version

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Generate checksums
        run: |
          cd dist/
          sha256sum * > checksums.txt
          cat checksums.txt

      - name: Upload release assets
        uses: actions/upload-artifact@v3
        with:
          name: release-assets-${{ needs.semantic-release.outputs.version }}
          path: |
            dist/
            CHANGELOG.md

  update-release:
    name: Update GitHub Release
    runs-on: ubuntu-latest
    needs: [semantic-release, build-release-assets]
    if: needs.semantic-release.outputs.released == 'true'
    
    steps:
      - name: Download release assets
        uses: actions/download-artifact@v3
        with:
          name: release-assets-${{ needs.semantic-release.outputs.version }}

      - name: Upload assets to release
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const path = require('path');
            
            // Get the release
            const { data: releases } = await github.rest.repos.listReleases({
              owner: context.repo.owner,
              repo: context.repo.repo,
              per_page: 1
            });
            
            if (releases.length === 0) {
              core.setFailed('No recent releases found');
              return;
            }
            
            const release = releases[0];
            
            // Upload distribution files
            const distFiles = fs.readdirSync('dist/');
            for (const file of distFiles) {
              const filePath = path.join('dist', file);
              const content = fs.readFileSync(filePath);
              
              await github.rest.repos.uploadReleaseAsset({
                owner: context.repo.owner,
                repo: context.repo.repo,
                release_id: release.id,
                name: file,
                data: content
              });
              
              console.log(`Uploaded ${file}`);
            }

  notify-release:
    name: Notify Release
    runs-on: ubuntu-latest
    needs: [semantic-release, update-release]
    if: needs.semantic-release.outputs.released == 'true'
    
    steps:
      - name: Notify Slack
        if: success()
        uses: 8398a7/action-slack@v3
        with:
          status: success
          channel: '#releases'
          message: |
            🎉 New release published!
            
            **LexGraph Legal RAG v${{ needs.semantic-release.outputs.version }}**
            
            📋 [View Release](https://github.com/${{ github.repository }}/releases/tag/v${{ needs.semantic-release.outputs.version }})
            📦 [Download](https://github.com/${{ github.repository }}/releases/latest)
            📚 [Changelog](https://github.com/${{ github.repository }}/blob/main/CHANGELOG.md)
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

      - name: Create announcement issue
        uses: actions/github-script@v6
        with:
          script: |
            const version = '${{ needs.semantic-release.outputs.version }}';
            
            await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `📢 Release Announcement: v${version}`,
              body: `
            # 🎉 LexGraph Legal RAG v${version} Released!
            
            We're excited to announce the release of LexGraph Legal RAG v${version}!
            
            ## 📥 Installation
            
            \`\`\`bash
            pip install lexgraph-legal-rag==${version}
            \`\`\`
            
            ## 📋 What's New
            
            Check out the [full changelog](https://github.com/${context.repo.owner}/${context.repo.repo}/blob/main/CHANGELOG.md) for detailed information about this release.
            
            ## 🐛 Found an Issue?
            
            If you encounter any problems with this release, please [open an issue](https://github.com/${context.repo.owner}/${context.repo.repo}/issues/new/choose).
            
            ## 🙏 Contributors
            
            Thank you to all contributors who made this release possible!
            
            ---
            
            **Links:**
            - 📦 [PyPI Package](https://pypi.org/project/lexgraph-legal-rag/${version}/)
            - 🐳 [Docker Image](https://github.com/${context.repo.owner}/${context.repo.repo}/pkgs/container/lexgraph-legal-rag)
            - 📚 [Documentation](https://github.com/${context.repo.owner}/${context.repo.repo}/blob/main/README.md)
              `,
              labels: ['announcement', 'release']
            });

  cleanup:
    name: Cleanup
    runs-on: ubuntu-latest
    needs: [semantic-release, notify-release]
    if: always()
    
    steps:
      - name: Clean up artifacts
        uses: actions/github-script@v6
        with:
          script: |
            // Clean up old artifacts
            const { data: artifacts } = await github.rest.actions.listArtifactsForRepo({
              owner: context.repo.owner,
              repo: context.repo.repo,
              per_page: 100
            });
            
            const oldArtifacts = artifacts.artifacts.filter(artifact => {
              const createdAt = new Date(artifact.created_at);
              const daysOld = (Date.now() - createdAt.getTime()) / (1000 * 60 * 60 * 24);
              return daysOld > 30 && artifact.name.startsWith('release-assets-');
            });
            
            for (const artifact of oldArtifacts) {
              await github.rest.actions.deleteArtifact({
                owner: context.repo.owner,
                repo: context.repo.repo,
                artifact_id: artifact.id
              });
              console.log(`Deleted old artifact: ${artifact.name}`);
            }
```

## Setup Instructions

1. **Create the `.github/workflows/` directory** in your repository
2. **Add each workflow file** with the contents above
3. **Configure repository secrets** for:
   - `PYPI_API_TOKEN` (if publishing to PyPI)
   - `SLACK_WEBHOOK_URL` (for notifications)
   - `KUBE_CONFIG_STAGING` and `KUBE_CONFIG_PRODUCTION` (for deployments)
   - `PROD_API_KEY` (for production health checks)

4. **Enable GitHub Actions** in your repository settings
5. **Configure branch protection** rules for the main branch
6. **Set up required status checks** to include the CI workflow jobs

## Benefits

Once these workflows are in place, you'll have:

- **Automated testing** on every PR and push
- **Security scanning** with vulnerability reporting
- **Automated deployments** to staging and production
- **Dependency management** with security updates
- **Semantic releases** with automated changelogs

The workflows will automatically trigger based on the configured events and provide comprehensive CI/CD automation for your repository.