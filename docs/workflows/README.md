# Workflow Requirements

## Overview

This directory contains workflow YAML files that require manual setup in GitHub Actions.

## Required Manual Setup

### 1. Repository Settings
- Enable GitHub Actions in repository settings
- Configure branch protection rules for `main` branch
- Set up required status checks

### 2. Secrets Configuration
Required secrets (configure in repository settings):
- `PYPI_API_TOKEN` - For package publishing
- `DOCKER_HUB_TOKEN` - For Docker image builds
- `CODECOV_TOKEN` - For coverage reporting

### 3. Workflow Files
Located in this directory:
- `ci.yml` - Continuous Integration pipeline
- `cd.yml` - Continuous Deployment pipeline  
- `security.yml` - Security scanning
- `dependency-update.yml` - Automated dependency updates
- `release.yml` - Release management
- `codeql-config.yml` - CodeQL security analysis

### 4. GitHub Features
Enable these features in repository settings:
- Issues and pull requests
- Discussions (optional)
- Security advisories
- Dependabot alerts

## Documentation

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides)

## Status

All workflow files are documented and ready for manual deployment to `.github/workflows/`.