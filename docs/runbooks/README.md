# LexGraph Legal RAG - Operations Runbooks

This directory contains operational runbooks for the LexGraph Legal RAG system. These guides help operations teams respond to common scenarios and incidents.

## Available Runbooks

- [Health Check Troubleshooting](health-check-troubleshooting.md)
- [Performance Issues](performance-issues.md)
- [Index Corruption Recovery](index-corruption-recovery.md)
- [High Memory Usage](high-memory-usage.md)
- [API Rate Limiting](api-rate-limiting.md)
- [Database Connection Issues](database-connection-issues.md)
- [Deployment Rollback](deployment-rollback.md)

## Quick Reference

### Health Endpoints
- **Application Health**: `GET /health`
- **Readiness Check**: `GET /ready`
- **Metrics**: `GET /metrics`
- **API Status**: `GET /api/v1/status`

### Common Commands
```bash
# Check application status
curl -f http://localhost:8000/health

# View metrics
curl http://localhost:8000/metrics

# Check logs
docker logs lexgraph-api

# Restart services
docker-compose restart lexgraph-api

# Scale services
docker-compose up --scale lexgraph-api=3
```

### Emergency Contacts
- **On-call Engineer**: [Slack @oncall]
- **System Admin**: [Slack @sysadmin]
- **Security Team**: [security@terragon.ai]

## Incident Response Process

1. **Acknowledge** the incident in monitoring system
2. **Assess** the impact and severity
3. **Communicate** with stakeholders
4. **Investigate** using runbooks and logs
5. **Resolve** the issue
6. **Document** the incident and lessons learned