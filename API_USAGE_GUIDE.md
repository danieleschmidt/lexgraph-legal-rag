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

## Authentication

All requests must include the `X-API-Key` header. Set this header to the value of the `API_KEY` environment variable configured on the server.

Example:

```bash
curl -H "X-API-Key: mysecret" http://localhost:8000/v1/ping
```

## Rate Limiting

The API enforces a limit of 60 requests per minute per client. Exceeding this
limit returns HTTP 429.
