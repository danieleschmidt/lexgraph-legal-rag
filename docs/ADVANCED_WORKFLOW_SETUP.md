# 🚀 Advanced GitHub Workflows Setup Guide

This guide provides production-ready GitHub Actions workflows for the LexGraph Legal RAG repository. These workflows implement enterprise-grade CI/CD practices tailored for advanced Python projects.

## 📋 Prerequisites

- Repository admin access to enable workflows
- GitHub secrets configured (see [Secrets Setup](#secrets-setup))
- Branch protection rules configured for `main` and `develop`

## 🔧 Workflow Installation

### 1. Create Workflow Directory
```bash
mkdir -p .github/workflows
mkdir -p .github/codeql
```

### 2. CI/CD Workflow (`ci.yml`)

Create `.github/workflows/ci.yml`:

```yaml
name: 🚀 Continuous Integration

on:
  push:
    branches: [ main, develop ]
    paths-ignore:
      - '**.md'
      - 'docs/**'
  pull_request:
    branches: [ main, develop ]
    paths-ignore:
      - '**.md'
      - 'docs/**'
  workflow_dispatch:

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  pre-commit:
    name: 🔍 Pre-commit Hooks
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - uses: pre-commit/action@v3.0.0
        env:
          SKIP: no-commit-to-branch

  test-matrix:
    name: 🧪 Test Suite
    runs-on: ubuntu-latest
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
        test-type: [unit, integration]
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest-xdist pytest-benchmark
      
      - name: Run ${{ matrix.test-type }} tests
        run: |
          pytest tests/ -m ${{ matrix.test-type }} --cov=lexgraph_legal_rag \
            --cov-report=xml --cov-report=term-missing \
            --junit-xml=test-results-${{ matrix.python-version }}-${{ matrix.test-type }}.xml \
            -n auto --maxfail=3
      
      - name: Upload coverage to Codecov
        if: matrix.python-version == '3.11' && matrix.test-type == 'unit'
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

  security-audit:
    name: 🔒 Security Audit
    runs-on: ubuntu-latest
    timeout-minutes: 15
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install bandit[toml] safety
      
      - name: Run Bandit security scan
        run: |
          bandit -r src/ -f json -o bandit-report.json
          bandit -r src/ -f sarif -o bandit-results.sarif
      
      - name: Upload Bandit results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: bandit-results.sarif
      
      - name: Check dependencies for vulnerabilities
        run: safety check --json --output safety-report.json

  performance-test:
    name: ⚡ Performance Tests
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      - name: Install k6
        run: |
          sudo gpg -k
          sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6
      
      - name: Start test server
        run: |
          python -m uvicorn lexgraph_legal_rag.api:create_api --host 0.0.0.0 --port 8000 &
          sleep 10
      
      - name: Run performance tests
        run: |
          k6 run tests/performance/load-test.js
          k6 run tests/performance/stress-test.js

  build-artifacts:
    name: 🏗️ Build Artifacts
    runs-on: ubuntu-latest
    timeout-minutes: 15
    needs: [pre-commit, test-matrix]
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Check package integrity
        run: twine check dist/*
      
      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: python-package-distributions
          path: dist/
          retention-days: 30

  docker-build:
    name: 🐳 Docker Build
    runs-on: ubuntu-latest
    timeout-minutes: 20
    needs: [pre-commit, test-matrix]
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          tags: lexgraph-legal-rag:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64

  mutation-testing:
    name: 🧬 Mutation Testing
    runs-on: ubuntu-latest
    timeout-minutes: 45
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install mutmut
      
      - name: Run mutation tests
        run: |
          mutmut run --paths-to-mutate=src/ --tests-dir=tests/ \
            --runner="python -m pytest -x" --CI
          mutmut junitxml > mutation-results.xml
      
      - name: Upload mutation test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: mutation-test-results
          path: mutation-results.xml
```

### 3. Security Workflow (`security.yml`)

Create `.github/workflows/security.yml`:

```yaml
name: 🔒 Security Scanning

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM UTC
  workflow_dispatch:

env:
  PYTHON_VERSION: '3.11'

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  codeql:
    name: 🔍 CodeQL Analysis
    runs-on: ubuntu-latest
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix:
        language: [ 'python', 'javascript' ]
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: ${{ matrix.language }}
          config-file: ./.github/codeql/codeql-config.yml

      - name: Set up Python
        if: matrix.language == 'python'
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install Python dependencies
        if: matrix.language == 'python'
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Autobuild
        uses: github/codeql-action/autobuild@v2

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
        with:
          category: "/language:${{matrix.language}}"

  dependency-scan:
    name: 🔍 Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install safety pip-audit

      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json
          safety check --full-report

      - name: Run pip-audit
        run: |
          pip-audit --format=json --output=pip-audit-report.json
          pip-audit --format=cyclonedx-json --output=sbom.json

      - name: Upload dependency scan results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: dependency-scan-results
          path: |
            safety-report.json
            pip-audit-report.json
            sbom.json

  container-scan:
    name: 🐳 Container Security Scan
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          load: true
          tags: lexgraph-legal-rag:security-scan
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: lexgraph-legal-rag:security-scan
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

      - name: Run Trivy filesystem scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-fs-results.sarif'

      - name: Upload Trivy filesystem scan results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-fs-results.sarif'

  secret-scan:
    name: 🔑 Secret Detection
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install detect-secrets
        run: pip install detect-secrets

      - name: Run detect-secrets scan
        run: |
          detect-secrets scan --all-files --baseline .secrets.baseline \
            --exclude-files '.*\.lock$' \
            --exclude-files '.*package-lock\.json$' \
            --exclude-files '.*\.min\.js$'

      - name: Verify secrets baseline
        run: |
          detect-secrets audit .secrets.baseline --report \
            --fail-on-unaudited --fail-on-live

  sbom-generation:
    name: 📋 SBOM Generation
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install cyclonedx-bom pip-licenses

      - name: Generate Python SBOM
        run: |
          cyclonedx-py --format json --output-file python-sbom.json
          cyclonedx-py --format xml --output-file python-sbom.xml

      - name: Generate license report
        run: |
          pip-licenses --format=json --output-file=licenses.json
          pip-licenses --format=csv --output-file=licenses.csv

      - name: Upload SBOM artifacts
        uses: actions/upload-artifact@v3
        with:
          name: sbom-artifacts
          path: |
            python-sbom.json
            python-sbom.xml
            licenses.json
            licenses.csv

  compliance-check:
    name: 📊 Compliance Validation
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4

      - name: Check license compliance
        run: |
          # Verify all source files have proper license headers
          find src/ -name "*.py" | xargs grep -L "Copyright\|License" || true
          
          # Check for required files
          test -f LICENSE || echo "❌ LICENSE file missing"
          test -f SECURITY.md || echo "❌ SECURITY.md file missing" 
          test -f CODE_OF_CONDUCT.md || echo "❌ CODE_OF_CONDUCT.md file missing"

      - name: Validate security policy
        run: |
          # Ensure security policy exists and is comprehensive
          grep -q "security@terragon" SECURITY.md || echo "❌ Security contact missing"
          grep -q "vulnerability" SECURITY.md || echo "❌ Vulnerability reporting missing"

  security-report:
    name: 📈 Security Report Summary
    runs-on: ubuntu-latest
    needs: [codeql, dependency-scan, container-scan, secret-scan, compliance-check]
    if: always()
    steps:
      - name: Generate security summary
        run: |
          echo "## 🔒 Security Scan Summary" >> $GITHUB_STEP_SUMMARY
          echo "| Check | Status |" >> $GITHUB_STEP_SUMMARY
          echo "|-------|--------|" >> $GITHUB_STEP_SUMMARY
          echo "| CodeQL Analysis | ${{ needs.codeql.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Dependency Scan | ${{ needs.dependency-scan.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Container Scan | ${{ needs.container-scan.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Secret Detection | ${{ needs.secret-scan.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Compliance Check | ${{ needs.compliance-check.result }} |" >> $GITHUB_STEP_SUMMARY
```

### 4. Release Workflow (`release.yml`)

Create `.github/workflows/release.yml`:

```yaml
name: 🚀 Release

on:
  push:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      release_type:
        description: 'Release type'
        required: true
        default: 'auto'
        type: choice
        options:
          - auto
          - patch
          - minor
          - major
          - prerelease

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

permissions:
  contents: write
  packages: write
  pull-requests: read

jobs:
  test-release:
    name: 🧪 Test Before Release
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest-xdist

      - name: Run critical tests
        run: |
          pytest tests/ -m "not slow" --cov=lexgraph_legal_rag \
            --cov-fail-under=80 -n auto --maxfail=1

      - name: Build and test package
        run: |
          python -m pip install build twine
          python -m build
          twine check dist/*

  security-check:
    name: 🔒 Security Validation
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install security tools
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install bandit safety

      - name: Run security scans
        run: |
          bandit -r src/ -ll
          safety check

  semantic-release:
    name: 📦 Semantic Release
    runs-on: ubuntu-latest
    timeout-minutes: 20
    needs: [test-release, security-check]
    outputs:
      released: ${{ steps.release.outputs.released }}
      version: ${{ steps.release.outputs.version }}
      upload_url: ${{ steps.release.outputs.upload_url }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Install semantic-release
        run: npm install

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Configure git
        run: |
          git config --global user.name 'github-actions[bot]'
          git config --global user.email 'github-actions[bot]@users.noreply.github.com'

      - name: Run semantic release
        id: release
        run: |
          if [ "${{ github.event.inputs.release_type }}" != "auto" ] && [ "${{ github.event.inputs.release_type }}" != "" ]; then
            echo "Manual release type: ${{ github.event.inputs.release_type }}"
            npm run release -- --release-as=${{ github.event.inputs.release_type }}
          else
            npm run release
          fi
          
          # Check if release was created
          if git describe --exact-match --tags HEAD >/dev/null 2>&1; then
            echo "released=true" >> $GITHUB_OUTPUT
            echo "version=$(git describe --tags --abbrev=0)" >> $GITHUB_OUTPUT
          else
            echo "released=false" >> $GITHUB_OUTPUT
          fi
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  build-python-package:
    name: 🐍 Build Python Package
    runs-on: ubuntu-latest
    timeout-minutes: 15
    needs: [semantic-release]
    if: needs.semantic-release.outputs.released == 'true'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install build dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Update version in pyproject.toml
        run: |
          VERSION="${{ needs.semantic-release.outputs.version }}"
          VERSION_CLEAN=${VERSION#v}  # Remove 'v' prefix if present
          sed -i "s/version = \".*\"/version = \"$VERSION_CLEAN\"/" pyproject.toml

      - name: Build package
        run: |
          python -m build
          twine check dist/*

      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: python-package-${{ needs.semantic-release.outputs.version }}
          path: dist/
          retention-days: 90

  build-docker-images:
    name: 🐳 Build Docker Images
    runs-on: ubuntu-latest
    timeout-minutes: 25
    needs: [semantic-release]
    if: needs.semantic-release.outputs.released == 'true'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: |
            ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}},value=${{ needs.semantic-release.outputs.version }}
            type=semver,pattern={{major}}.{{minor}},value=${{ needs.semantic-release.outputs.version }}
            type=semver,pattern={{major}},value=${{ needs.semantic-release.outputs.version }}
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker images
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          platforms: linux/amd64,linux/arm64
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            VERSION=${{ needs.semantic-release.outputs.version }}

  publish-pypi:
    name: 📦 Publish to PyPI
    runs-on: ubuntu-latest
    timeout-minutes: 10
    needs: [build-python-package, semantic-release]
    if: needs.semantic-release.outputs.released == 'true'
    environment:
      name: pypi
      url: https://pypi.org/p/lexgraph-legal-rag
    permissions:
      id-token: write
    steps:
      - name: Download build artifacts
        uses: actions/download-artifact@v3
        with:
          name: python-package-${{ needs.semantic-release.outputs.version }}
          path: dist/

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          verbose: true

  create-github-release:
    name: 📋 Create GitHub Release
    runs-on: ubuntu-latest
    timeout-minutes: 10
    needs: [semantic-release, build-python-package, build-docker-images]
    if: needs.semantic-release.outputs.released == 'true'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Download build artifacts
        uses: actions/download-artifact@v3
        with:
          name: python-package-${{ needs.semantic-release.outputs.version }}
          path: dist/

      - name: Generate release notes
        id: release_notes
        run: |
          VERSION="${{ needs.semantic-release.outputs.version }}"
          
          # Get the changelog for this version
          if [ -f CHANGELOG.md ]; then
            CHANGELOG=$(awk "/^## \[?${VERSION#v}/ {flag=1; next} /^## / && flag {exit} flag" CHANGELOG.md)
            echo "changelog<<EOF" >> $GITHUB_OUTPUT
            echo "$CHANGELOG" >> $GITHUB_OUTPUT
            echo "EOF" >> $GITHUB_OUTPUT
          else
            echo "changelog=Release $VERSION" >> $GITHUB_OUTPUT
          fi

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: ${{ needs.semantic-release.outputs.version }}
          name: Release ${{ needs.semantic-release.outputs.version }}
          body: |
            ## 🚀 What's New
            
            ${{ steps.release_notes.outputs.changelog }}
            
            ## 📦 Installation
            
            ```bash
            pip install lexgraph-legal-rag==${{ needs.semantic-release.outputs.version }}
            ```
            
            ## 🐳 Docker
            
            ```bash
            docker pull ghcr.io/${{ github.repository }}:${{ needs.semantic-release.outputs.version }}
            ```
            
            ## 🔗 Links
            
            - [PyPI Package](https://pypi.org/project/lexgraph-legal-rag/${{ needs.semantic-release.outputs.version }}/)
            - [Docker Images](https://github.com/${{ github.repository }}/pkgs/container/lexgraph-legal-rag)
            - [Documentation](https://github.com/${{ github.repository }}#readme)
          files: |
            dist/*
          draft: false
          prerelease: false

  notify-deployment:
    name: 📢 Deployment Notification
    runs-on: ubuntu-latest
    needs: [semantic-release, publish-pypi, create-github-release]
    if: needs.semantic-release.outputs.released == 'true'
    steps:
      - name: Notify successful release
        run: |
          echo "🎉 Successfully released version ${{ needs.semantic-release.outputs.version }}"
          echo "📦 PyPI: https://pypi.org/project/lexgraph-legal-rag/${{ needs.semantic-release.outputs.version }}/"
          echo "🐳 Docker: ghcr.io/${{ github.repository }}:${{ needs.semantic-release.outputs.version }}"
          echo "📋 Release: https://github.com/${{ github.repository }}/releases/tag/${{ needs.semantic-release.outputs.version }}"
```

### 5. CodeQL Configuration (`codeql-config.yml`)

Create `.github/codeql/codeql-config.yml`:

```yaml
name: "CodeQL Configuration for LexGraph Legal RAG"

# Paths to exclude from analysis
paths-ignore:
  - "tests/**"
  - "docs/**"
  - "**/*.md"
  - "**/*.json"
  - "**/*.yaml"
  - "**/*.yml"
  - ".github/**"
  - "monitoring/**"
  - "k8s/**"

# Paths to include (override exclude patterns if needed)
paths:
  - "src/**"

# Query filters to use
queries:
  - uses: security-extended
  - uses: security-and-quality

# Configure specific query suites
query-filters:
  - exclude:
      id: py/unused-import
  - exclude:
      id: py/similar-function
  - include:
      severity: error
  - include:
      severity: warning
      security-severity: high

# Disable default queries that may be noisy
disable-default-queries: false

# Configure external library support
external-repository-token: ${{ secrets.GITHUB_TOKEN }}

# Analysis configuration
packs:
  python:
    - codeql/python-queries
    - codeql/python-security-queries
```

## 🔐 Secrets Setup

Configure the following secrets in your repository settings:

### Required Secrets
- `GITHUB_TOKEN` (automatically provided)
- `CODECOV_TOKEN` (optional, for coverage reporting)

### Optional Secrets for Enhanced Features
- `PYPI_API_TOKEN` (for PyPI publishing)
- `DOCKER_REGISTRY_TOKEN` (for custom registries)
- `SLACK_WEBHOOK_URL` (for notifications)

## 🚀 Activation Steps

1. **Copy Workflow Files**: Create the above files in your repository
2. **Enable Actions**: Go to Settings → Actions → General → Allow all actions
3. **Configure Branch Protection**: 
   - Require status checks for PRs
   - Require branches to be up to date
   - Include administrators in restrictions
4. **Set up Environments**: Create `pypi` environment for production deployments
5. **Test Workflows**: Create a test PR to verify all workflows run correctly

## 📊 Features Overview

### CI Workflow Features
- ✅ Multi-Python version testing (3.8-3.12)
- ✅ Pre-commit hook validation
- ✅ Security scanning with Bandit
- ✅ Performance testing with k6
- ✅ Mutation testing for PR validation
- ✅ Docker multi-platform builds
- ✅ Artifact generation and storage

### Security Workflow Features
- ✅ CodeQL static analysis
- ✅ Dependency vulnerability scanning
- ✅ Container security scanning with Trivy
- ✅ Secret detection with baseline
- ✅ SBOM generation for supply chain security
- ✅ Compliance validation

### Release Workflow Features
- ✅ Semantic versioning automation
- ✅ PyPI package publishing
- ✅ GitHub Container Registry publishing
- ✅ Multi-architecture Docker builds
- ✅ Automated release notes generation
- ✅ GitHub Release creation with artifacts

## 🔧 Customization

### Environment Variables
Modify workflow environment variables as needed:
- `PYTHON_VERSION`: Default Python version
- `NODE_VERSION`: Node.js version for semantic-release

### Test Configuration
Adjust test commands and coverage thresholds in the workflows to match your project requirements.

### Security Configuration
Customize security scanning rules and SARIF upload settings based on your organization's security policies.

This setup provides enterprise-grade CI/CD automation tailored for advanced Python projects with comprehensive security, testing, and deployment capabilities.