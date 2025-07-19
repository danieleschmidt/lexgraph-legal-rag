# LexGraph Legal RAG API Usage Guide

This document explains how to interact with the LexGraph Legal RAG API, including authentication, versioning, and key endpoints.

## Quick Start

1. **Set your API key** as an environment variable or obtain one from your administrator
2. **Choose your API version** using one of the negotiation methods below
3. **Make requests** to endpoints with proper authentication headers

## API Versioning & Negotiation

The LexGraph Legal RAG API supports multiple versions with flexible version negotiation. You can specify your desired API version using any of these methods:

### Supported Versions
- **v1**: Stable version with core functionality
- **v2**: Enhanced version with metadata wrappers and improved error handling

### Version Negotiation Methods

#### 1. Accept Header (Recommended)
```bash
curl -H "Accept: application/vnd.lexgraph.v1+json" \
     -H "X-API-Key: your-api-key" \
     http://localhost:8000/ping
```

#### 2. URL Path
```bash
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/v1/ping
```

#### 3. Query Parameter  
```bash
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/ping?version=v1
```

#### 4. Custom Header
```bash
curl -H "X-API-Version: v1" \
     -H "X-API-Key: your-api-key" \
     http://localhost:8000/ping
```

### Default Version Fallback

If no version is specified, the API defaults to **v1**. For production applications, explicitly specify the version to ensure compatibility.

## Authentication

All API endpoints require authentication using an API key.

### Headers Required
- `X-API-Key`: Your API key (required for all endpoints except `/health` and `/ready`)

### Example Request
```bash
curl -H "X-API-Key: your-secret-api-key" \
     -H "Content-Type: application/json" \
     http://localhost:8000/v1/ping
```

### Response Headers
The API includes helpful headers in responses:
- `X-API-Version`: The negotiated API version used for the request
- `X-Supported-Versions`: List of all supported API versions

## Core Endpoints

### Health & Status

#### Health Check
```bash
GET /health
```
Returns basic service health status (no authentication required).

#### Readiness Check  
```bash
GET /ready
```
Returns detailed readiness status including cache, memory, and external services (no authentication required).

#### Version Information
```bash
GET /version
```
Returns API version information and negotiation details.

### Testing & Utilities

#### Ping
```bash
GET /v1/ping
```
Basic connectivity test.

**Response (v1):**
```json
{
  "version": "v1",
  "ping": "pong", 
  "timestamp": 1234567890.123
}
```

#### Addition
```bash
GET /v1/add?a=5&b=7
```
Simple arithmetic endpoint for testing.

**Response (v1):**
```json
{
  "result": 12
}
```

**Response (v2):**
```json
{
  "success": true,
  "data": {
    "result": 12
  },
  "metadata": {
    "version": "v2",
    "timestamp": 1234567890.123
  }
}
```

## Administrative Endpoints

All admin endpoints require authentication and are prefixed with `/admin/`.

### API Key Management

#### Rotate Keys
```bash
POST /v1/admin/rotate-keys
Content-Type: application/json

{
  "new_primary_key": "new-secure-api-key-123"
}
```

#### Revoke Key
```bash
POST /v1/admin/revoke-key
Content-Type: application/json

{
  "api_key": "key-to-revoke"
}
```

#### Key Status
```bash
GET /v1/admin/key-status
```

### Monitoring

#### Metrics Summary
```bash
GET /v1/admin/metrics
```
Returns comprehensive application metrics including cache statistics, memory usage, and performance data.

## Rate Limiting

The API enforces rate limiting to ensure fair usage:

- **Default Limit**: 60 requests per minute per client
- **Rate Limit Headers**: Check response headers for current limits
- **429 Status**: Returned when rate limit is exceeded

### Rate Limit Headers
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1234567890
```

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (missing or invalid API key)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error

### Error Response Format

**v1 Format:**
```json
{
  "detail": "Error message"
}
```

**v2 Format:**
```json
{
  "success": false,
  "error": {
    "detail": "Error message",
    "code": "ERROR_CODE"
  },
  "metadata": {
    "version": "v2",
    "timestamp": 1234567890.123
  }
}
```

## Interactive Documentation

The API provides interactive documentation at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## Best Practices

1. **Version Pinning**: Always specify API version in production
2. **Error Handling**: Implement retry logic with exponential backoff
3. **Rate Limits**: Respect rate limits and implement client-side throttling
4. **API Keys**: Keep API keys secure and rotate them regularly
5. **Monitoring**: Monitor your usage via the metrics endpoints

## Examples

### Python with requests
```python
import requests

headers = {
    'X-API-Key': 'your-api-key',
    'Accept': 'application/vnd.lexgraph.v1+json'
}

response = requests.get('http://localhost:8000/ping', headers=headers)
print(response.json())
```

### JavaScript with fetch
```javascript
const headers = {
    'X-API-Key': 'your-api-key',
    'Accept': 'application/vnd.lexgraph.v1+json'
};

fetch('http://localhost:8000/ping', { headers })
    .then(response => response.json())
    .then(data => console.log(data));
```

## Support

For additional support or questions about the API:
- Check the interactive documentation at `/docs`
- Review the health check endpoints for service status
- Contact support at support@lexgraph.com
