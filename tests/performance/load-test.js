import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

// Test configuration
export const options = {
  stages: [
    { duration: '1m', target: 10 },   // Ramp up to 10 users
    { duration: '3m', target: 10 },   // Stay at 10 users
    { duration: '1m', target: 20 },   // Ramp up to 20 users
    { duration: '3m', target: 20 },   // Stay at 20 users
    { duration: '1m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests must complete below 500ms
    http_req_failed: ['rate<0.01'],   // Error rate must be below 1%
    errors: ['rate<0.01'],            // Custom error rate
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'perf-test-key';

const headers = {
  'X-API-Key': API_KEY,
  'Content-Type': 'application/json',
  'Accept': 'application/json',
};

export default function () {
  // Test health endpoint (no auth required)
  let response = http.get(`${BASE_URL}/health`);
  check(response, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 100ms': (r) => r.timings.duration < 100,
  });

  sleep(0.1);

  // Test ping endpoint
  response = http.get(`${BASE_URL}/v1/ping`, { headers });
  check(response, {
    'ping status is 200': (r) => r.status === 200,
    'ping response contains pong': (r) => r.json().ping === 'pong',
    'ping response time < 200ms': (r) => r.timings.duration < 200,
  });

  errorRate.add(response.status !== 200);
  responseTime.add(response.timings.duration);

  sleep(0.1);

  // Test add endpoint with random values
  const a = Math.floor(Math.random() * 100);
  const b = Math.floor(Math.random() * 100);
  
  response = http.get(`${BASE_URL}/v1/add?a=${a}&b=${b}`, { headers });
  check(response, {
    'add status is 200': (r) => r.status === 200,
    'add result is correct': (r) => r.json().result === a + b,
    'add response time < 300ms': (r) => r.timings.duration < 300,
  });

  errorRate.add(response.status !== 200);
  responseTime.add(response.timings.duration);

  sleep(0.1);

  // Test version endpoint
  response = http.get(`${BASE_URL}/version`, { headers });
  check(response, {
    'version status is 200': (r) => r.status === 200,
    'version response has supported_versions': (r) => r.json().supported_versions,
    'version response time < 150ms': (r) => r.timings.duration < 150,
  });

  errorRate.add(response.status !== 200);
  responseTime.add(response.timings.duration);

  sleep(0.5);
}

export function handleSummary(data) {
  return {
    'performance-report.json': JSON.stringify(data, null, 2),
    'performance-summary.html': htmlReport(data),
  };
}

function htmlReport(data) {
  return `
<!DOCTYPE html>
<html>
<head>
    <title>LexGraph Legal RAG Performance Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; }
        .metric { margin: 10px 0; padding: 10px; border-left: 4px solid #007cba; }
        .pass { border-left-color: #28a745; }
        .fail { border-left-color: #dc3545; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>LexGraph Legal RAG Performance Test Report</h1>
        <p>Generated: ${new Date().toISOString()}</p>
        <p>Duration: ${data.state.testRunDurationMs / 1000}s</p>
        <p>Virtual Users: ${data.options.stages ? data.options.stages.map(s => s.target).join(' → ') : 'N/A'}</p>
    </div>

    <h2>Key Metrics</h2>
    <div class="metric ${data.metrics.http_req_duration.values.p95 < 500 ? 'pass' : 'fail'}">
        <strong>95th Percentile Response Time:</strong> ${data.metrics.http_req_duration.values.p95.toFixed(2)}ms
        (Threshold: < 500ms)
    </div>
    
    <div class="metric ${data.metrics.http_req_failed.values.rate < 0.01 ? 'pass' : 'fail'}">
        <strong>Error Rate:</strong> ${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%
        (Threshold: < 1%)
    </div>

    <div class="metric">
        <strong>Total Requests:</strong> ${data.metrics.http_reqs.values.count}
    </div>

    <div class="metric">
        <strong>Requests per Second:</strong> ${data.metrics.http_reqs.values.rate.toFixed(2)}
    </div>

    <h2>Response Time Distribution</h2>
    <table>
        <tr><th>Percentile</th><th>Response Time (ms)</th></tr>
        <tr><td>50th (Median)</td><td>${data.metrics.http_req_duration.values.p50.toFixed(2)}</td></tr>
        <tr><td>90th</td><td>${data.metrics.http_req_duration.values.p90.toFixed(2)}</td></tr>
        <tr><td>95th</td><td>${data.metrics.http_req_duration.values.p95.toFixed(2)}</td></tr>
        <tr><td>99th</td><td>${data.metrics.http_req_duration.values.p99.toFixed(2)}</td></tr>
    </table>

    <h2>HTTP Status Codes</h2>
    <table>
        <tr><th>Status Code</th><th>Count</th></tr>
        ${Object.entries(data.metrics.http_req_failed.values)
          .map(([key, value]) => `<tr><td>${key}</td><td>${value}</td></tr>`)
          .join('')}
    </table>
</body>
</html>`;
}