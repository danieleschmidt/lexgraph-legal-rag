# LexGraph Legal RAG API Guide

This comprehensive guide covers all aspects of using the LexGraph Legal RAG API, from basic setup to advanced usage patterns.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Formats](#requestresponse-formats)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Versioning](#versioning)
8. [Examples](#examples)
9. [SDKs and Libraries](#sdks-and-libraries)
10. [Troubleshooting](#troubleshooting)

## Quick Start

### Base URL

```
Production: https://api.lexgraph.terragon.ai
Staging:    https://staging-api.lexgraph.terragon.ai
Local:      http://localhost:8000
```

### Simple Example

```bash
curl -X GET "https://api.lexgraph.terragon.ai/v1/ping" \
  -H "X-API-Key: your-api-key-here" \
  -H "Accept: application/json"
```

## Authentication

### API Key Authentication

All API requests require an API key provided in the `X-API-Key` header:

```http
X-API-Key: your-api-key-here
```

### Getting an API Key

1. Sign up at [lexgraph.terragon.ai](https://lexgraph.terragon.ai)
2. Navigate to your dashboard
3. Generate a new API key
4. Store it securely

### Security Best Practices

- Never expose API keys in client-side code
- Use environment variables to store keys
- Rotate keys regularly
- Use different keys for different environments

## API Endpoints

### Health and Status

#### Health Check

```http
GET /health
```

Returns basic health status of the service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1640995200.0,
  "version": "1.0.0",
  "uptime_seconds": 3600.5
}
```

#### Detailed Health Check

```http
GET /health/detailed
```

Returns comprehensive health information including component status.

#### Readiness Probe

```http
GET /ready
```

Kubernetes readiness probe endpoint.

#### Liveness Probe

```http
GET /live
```

Kubernetes liveness probe endpoint.

### Core API

#### Ping

```http
GET /v1/ping
```

Simple connectivity test.

**Response:**
```json
{
  "ping": "pong",
  "timestamp": 1640995200.0,
  "version": "1.0.0"
}
```

#### Add Numbers (Example)

```http
GET /v1/add?a={number}&b={number}
```

Example endpoint for testing.

**Parameters:**
- `a` (required): First number
- `b` (required): Second number

**Response:**
```json
{
  "result": 15,
  "operation": "addition",
  "inputs": {"a": 7, "b": 8}
}
```

### Legal Document Analysis

#### Document Search

```http
POST /v1/search
```

Search legal documents using semantic similarity.

**Request Body:**
```json
{
  "query": "indemnification clauses in commercial contracts",
  "limit": 10,
  "threshold": 0.7,
  "filters": {
    "document_type": "contract",
    "jurisdiction": "US"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "document_id": "doc_123",
      "title": "Commercial Services Agreement",
      "similarity_score": 0.89,
      "excerpt": "Party A shall indemnify...",
      "metadata": {
        "jurisdiction": "US",
        "document_type": "contract",
        "date_created": "2023-01-15"
      }
    }
  ],
  "total_results": 1,
  "query_time_ms": 245
}
```

#### Document Analysis

```http
POST /v1/analyze
```

Analyze legal documents using multi-agent reasoning.

**Request Body:**
```json
{
  "document_text": "The licensee shall...",
  "analysis_type": "clause_extraction",
  "options": {
    "include_citations": true,
    "confidence_threshold": 0.8
  }
}
```

#### Clause Explanation

```http
POST /v1/explain
```

Get detailed explanations of legal clauses.

**Request Body:**
```json
{
  "clause_text": "Force majeure clause text...",
  "context": "commercial_contract",
  "jurisdiction": "US"
}
```

### Version Information

#### Get Supported Versions

```http
GET /version
```

Returns information about supported API versions.

**Response:**
```json
{
  "current_version": "1.0",
  "supported_versions": ["1.0"],
  "deprecated_versions": [],
  "version_info": {
    "1.0": {
      "status": "stable",
      "release_date": "2024-01-01",
      "deprecation_date": null
    }
  }
}
```

## Request/Response Formats

### Content Types

- Request: `application/json`
- Response: `application/json`

### Common Headers

**Request Headers:**
```http
X-API-Key: your-api-key
Content-Type: application/json
Accept: application/json
X-API-Version: 1.0
X-Correlation-ID: optional-tracking-id
```

**Response Headers:**
```http
Content-Type: application/json
X-API-Version: 1.0
X-Correlation-ID: tracking-id
X-Rate-Limit-Remaining: 59
X-Rate-Limit-Reset: 1640995260
```

### Standard Response Format

All API responses follow a consistent format:

```json
{
  "data": {},
  "meta": {
    "version": "1.0",
    "timestamp": 1640995200.0,
    "correlation_id": "abc-123"
  }
}
```

## Error Handling

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "query",
      "issue": "Required field missing"
    }
  },
  "meta": {
    "version": "1.0",
    "timestamp": 1640995200.0,
    "correlation_id": "abc-123"
  }
}
```

### Common Error Codes

- `INVALID_API_KEY` - API key is missing or invalid
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `VALIDATION_ERROR` - Request validation failed
- `RESOURCE_NOT_FOUND` - Requested resource doesn't exist
- `SERVICE_UNAVAILABLE` - Service temporarily unavailable

## Rate Limiting

### Limits

- **Default**: 60 requests per minute per API key
- **Burst**: Up to 10 requests per second
- **Daily**: 10,000 requests per day

### Headers

Rate limit information is returned in response headers:

```http
X-Rate-Limit-Limit: 60
X-Rate-Limit-Remaining: 59
X-Rate-Limit-Reset: 1640995260
X-Rate-Limit-Window: 60
```

### Handling Rate Limits

```python
import time
import requests

def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code == 429:
            reset_time = int(response.headers.get('X-Rate-Limit-Reset', 0))
            wait_time = reset_time - int(time.time())
            if wait_time > 0:
                time.sleep(wait_time)
            continue
            
        return response
    
    raise Exception("Max retries exceeded")
```

## Versioning

### Version Header

Specify API version using the header:

```http
X-API-Version: 1.0
```

### URL Versioning

Alternatively, include version in the URL:

```http
/v1/search
/v1/analyze
```

### Deprecation Policy

- New versions are backward compatible when possible
- Deprecated versions are supported for 12 months
- Breaking changes trigger major version increments

## Examples

### Python

```python
import requests
import os

class LexGraphClient:
    def __init__(self, api_key, base_url="https://api.lexgraph.terragon.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def search_documents(self, query, limit=10):
        response = self.session.post(
            f"{self.base_url}/v1/search",
            json={"query": query, "limit": limit}
        )
        response.raise_for_status()
        return response.json()
    
    def analyze_document(self, text, analysis_type="clause_extraction"):
        response = self.session.post(
            f"{self.base_url}/v1/analyze",
            json={
                "document_text": text,
                "analysis_type": analysis_type
            }
        )
        response.raise_for_status()
        return response.json()

# Usage
client = LexGraphClient(os.getenv("LEXGRAPH_API_KEY"))
results = client.search_documents("indemnification clauses")
print(results)
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

class LexGraphClient {
    constructor(apiKey, baseUrl = 'https://api.lexgraph.terragon.ai') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            headers: {
                'X-API-Key': apiKey,
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
    }
    
    async searchDocuments(query, limit = 10) {
        try {
            const response = await this.client.post('/v1/search', {
                query,
                limit
            });
            return response.data;
        } catch (error) {
            throw new Error(`Search failed: ${error.response?.data?.error?.message}`);
        }
    }
    
    async analyzeDocument(text, analysisType = 'clause_extraction') {
        try {
            const response = await this.client.post('/v1/analyze', {
                document_text: text,
                analysis_type: analysisType
            });
            return response.data;
        } catch (error) {
            throw new Error(`Analysis failed: ${error.response?.data?.error?.message}`);
        }
    }
}

// Usage
const client = new LexGraphClient(process.env.LEXGRAPH_API_KEY);
client.searchDocuments('indemnification clauses')
    .then(results => console.log(results))
    .catch(error => console.error(error));
```

### cURL Examples

```bash
# Search documents
curl -X POST "https://api.lexgraph.terragon.ai/v1/search" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "indemnification clauses",
    "limit": 5
  }'

# Analyze document
curl -X POST "https://api.lexgraph.terragon.ai/v1/analyze" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "The licensee shall indemnify...",
    "analysis_type": "clause_extraction"
  }'

# Get version info
curl -X GET "https://api.lexgraph.terragon.ai/version" \
  -H "X-API-Key: your-api-key"
```

## SDKs and Libraries

### Official SDKs

- **Python**: `pip install lexgraph-python`
- **JavaScript**: `npm install lexgraph-js`
- **Go**: `go get github.com/terragon-labs/lexgraph-go`

### Community Libraries

- **Ruby**: lexgraph-ruby (community)
- **PHP**: lexgraph-php (community)
- **Java**: lexgraph-java (community)

## Troubleshooting

### Common Issues

#### Authentication Errors

**Problem**: `401 Unauthorized`

**Solutions**:
- Verify API key is correct
- Check API key hasn't expired
- Ensure API key is in `X-API-Key` header

#### Rate Limiting

**Problem**: `429 Too Many Requests`

**Solutions**:
- Implement exponential backoff
- Respect rate limit headers
- Consider upgrading API plan

#### Validation Errors

**Problem**: `422 Validation Error`

**Solutions**:
- Check request format matches API specification
- Verify all required fields are included
- Validate field types and formats

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your API calls here
```

### Support

- **Documentation**: [docs.lexgraph.terragon.ai](https://docs.lexgraph.terragon.ai)
- **Support Email**: support@terragon.ai
- **GitHub Issues**: [github.com/terragon-labs/lexgraph-legal-rag/issues](https://github.com/terragon-labs/lexgraph-legal-rag/issues)
- **Community Forum**: [community.terragon.ai](https://community.terragon.ai)

### API Status

Check service status at: [status.terragon.ai](https://status.terragon.ai)