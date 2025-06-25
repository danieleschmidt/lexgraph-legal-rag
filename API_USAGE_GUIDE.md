# API Usage Guide

This document explains how to interact with the LexGraph Legal RAG API.

## Versioned Endpoints

Each API endpoint is prefixed with a version string. The current supported
version is `v1`.

Example:

```bash
GET /v1/ping
```

This returns:

```json
{"version": "v1", "ping": "pong"}
```

## Default Version Fallback

If you omit the version prefix, the server assumes the latest version. For
example:

```bash
GET /ping
```

will return the same response as `/v1/ping`.

Use the versioned path if you need to pin to a specific API version.
