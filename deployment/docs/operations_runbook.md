# Operations Runbook - Bioneural Olfactory Fusion System

## Emergency Procedures

### Service Down

1. Check deployment status:
   ```bash
   kubectl get deployment bioneural-olfactory-fusion-api -n bioneural-olfactory-fusion-production
   ```

2. Check pod status:
   ```bash
   kubectl get pods -n bioneural-olfactory-fusion-production
   ```

3. Check logs:
   ```bash
   kubectl logs -f deployment/bioneural-olfactory-fusion-api -n bioneural-olfactory-fusion-production
   ```

4. If necessary, restart deployment:
   ```bash
   kubectl rollout restart deployment/bioneural-olfactory-fusion-api -n bioneural-olfactory-fusion-production
   ```

### High Load

1. Check current scaling:
   ```bash
   kubectl get hpa -n bioneural-olfactory-fusion-production
   ```

2. Manual scaling if needed:
   ```bash
   kubectl scale deployment bioneural-olfactory-fusion-api --replicas=10 -n bioneural-olfactory-fusion-production
   ```

3. Monitor metrics:
   ```bash
   kubectl top pods -n bioneural-olfactory-fusion-production
   ```

## Monitoring Checklist

- [ ] All pods running and ready
- [ ] Response times < 500ms (95th percentile)
- [ ] Error rate < 1%
- [ ] CPU usage < 70%
- [ ] Memory usage < 80%
- [ ] Cache hit rate > 80%

## Scheduled Maintenance

### Weekly Tasks
- Review error logs
- Check resource utilization trends
- Verify backup integrity
- Update security scan results

### Monthly Tasks
- Performance testing
- Capacity planning review
- Security audit
- Dependency updates

## Escalation Procedures

1. **Level 1**: Automated alerts and self-healing
2. **Level 2**: On-call engineer notification
3. **Level 3**: Team lead escalation
4. **Level 4**: Management notification for business impact
