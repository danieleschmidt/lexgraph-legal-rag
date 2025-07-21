# CORS Security Configuration

## Overview

The API implements secure CORS (Cross-Origin Resource Sharing) configuration with environment-based controls and different security levels for development vs production environments.

## Security Features

### Production Mode (Default)
- **Origins**: Empty list by default (no cross-origin requests allowed)
- **Methods**: Limited to GET, POST, OPTIONS only
- **Headers**: Restricted to essential headers only
- **Configuration**: Must be explicitly configured via environment variables

### Development/Test Mode
- **Origins**: Allows localhost development origins (3000, 8080)
- **Methods**: More permissive (GET, POST, PUT, DELETE, OPTIONS)
- **Headers**: Same restricted set as production

## Environment Configuration

### Required Variables (Production)
```bash
CORS_ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
```

### Optional Variables
```bash
CORS_ALLOWED_METHODS="GET,POST,OPTIONS"
CORS_ALLOWED_HEADERS="Accept,Content-Type,X-API-Key"
```

## Security Principles

1. **Deny by Default**: Production mode blocks all origins unless explicitly allowed
2. **Principle of Least Privilege**: Minimal methods and headers in production
3. **Environment Separation**: Different policies for dev vs prod
4. **Explicit Configuration**: No wildcard (*) origins in production
5. **Credential Protection**: Secure credential handling with origin restrictions

## Migration from Previous Configuration

### Before (Insecure)
```python
allow_origins=["*"]  # Allowed all origins - SECURITY RISK
```

### After (Secure)
```python
allow_origins=_get_cors_origins(test_mode)  # Environment-controlled, secure defaults
```

## Compliance

This configuration helps meet security standards:
- OWASP ASVS Level 2
- NIST Cybersecurity Framework
- SOC 2 Type II requirements
- General data protection regulations

## Testing

Comprehensive test coverage ensures:
- Production mode blocks unauthorized origins
- Development mode allows localhost
- Environment configuration works correctly
- Security headers are properly set
- Methods and headers are appropriately restricted