import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

// Stress test configuration - higher load
export const options = {
  stages: [
    { duration: '2m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 200 },  // Spike to 200 users
    { duration: '1m', target: 200 },  // Brief spike
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // Allow higher latency under stress
    http_req_failed: ['rate<0.05'],    // Allow 5% error rate under stress
    errors: ['rate<0.05'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'stress-test-key';

const headers = {
  'X-API-Key': API_KEY,
  'Content-Type': 'application/json',
  'Accept': 'application/json',
};

export default function () {
  // Mix of different endpoints to simulate real usage
  const endpoints = [
    () => http.get(`${BASE_URL}/health`),
    () => http.get(`${BASE_URL}/v1/ping`, { headers }),
    () => {
      const a = Math.floor(Math.random() * 1000);
      const b = Math.floor(Math.random() * 1000);
      return http.get(`${BASE_URL}/v1/add?a=${a}&b=${b}`, { headers });
    },
    () => http.get(`${BASE_URL}/version`, { headers }),
    () => http.get(`${BASE_URL}/ready`),
  ];

  // Randomly select an endpoint
  const randomEndpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
  const response = randomEndpoint();

  check(response, {
    'status is 200 or 401 (expected for unauth)': (r) => [200, 401].includes(r.status),
    'response time < 5000ms': (r) => r.timings.duration < 5000,
  });

  errorRate.add(![200, 401].includes(response.status));
  responseTime.add(response.timings.duration);

  // Shorter sleep for higher load
  sleep(Math.random() * 0.5);
}

export function handleSummary(data) {
  return {
    'stress-test-report.json': JSON.stringify(data, null, 2),
  };
}