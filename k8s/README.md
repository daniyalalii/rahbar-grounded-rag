# k8s/ — Kubernetes Manifests for RAHBAR RAGOps

## Quick Start (fresh cluster)

```bash
# 1. Start minikube
minikube start --memory=4096 --cpus=4

# 2. Build images inside minikube's Docker
eval $(minikube docker-env)
docker build -t rahbar-api:dev    ./api
docker build -t rahbar-indexer:dev ./indexer

# 3. Apply all manifests
kubectl apply -k k8s/base/

# 4. Wait for all pods
kubectl wait --for=condition=ready pod -l app=api -n rahbar --timeout=120s
```

---

## Commands for Each Part

### Part A — Data Verification + Incremental Update

```bash
# Run ingestion with verification (Batch B)
kubectl delete job rahbar-indexer -n rahbar 2>/dev/null; \
kubectl apply -f k8s/base/indexer-job.yaml -n rahbar
kubectl logs job/rahbar-indexer -n rahbar -f

# Second run on unchanged corpus (must show 0 new chunks)
kubectl delete job rahbar-indexer -n rahbar
kubectl apply -f k8s/base/indexer-job.yaml -n rahbar
kubectl logs job/rahbar-indexer -n rahbar -f
```

### Part B — Offline Evaluation (3 Levels)

```bash
# Run the evaluation Job
kubectl delete job rahbar-eval -n rahbar 2>/dev/null
kubectl apply -f k8s/base/eval-job.yaml
kubectl wait --for=condition=complete job/rahbar-eval -n rahbar --timeout=300s
kubectl logs job/rahbar-eval -n rahbar
```

### Part C — Release Gate + Controlled Rollout

```bash
# Deploy candidate
kubectl apply -f k8s/base/api-candidate.yaml

# Run eval against candidate
kubectl apply -f k8s/base/eval-job.yaml   # (after editing --api to point to api-candidate:8000)

# Gate FAIL: raise threshold so candidate fails
kubectl edit configmap ragops-gate-thresholds -n rahbar
# set RECALL_AT_5_MAX_DROP: "0.001"  → gate will fail

# Gate PASS: restore threshold
kubectl edit configmap ragops-gate-thresholds -n rahbar
# set RECALL_AT_5_MAX_DROP: "0.05"  → gate will pass

# If PASS: rolling update
kubectl set image deployment/api api=rahbar-api:candidate -n rahbar
kubectl rollout status deployment/api -n rahbar

# Rollback
kubectl rollout undo deployment/api -n rahbar
kubectl rollout status deployment/api -n rahbar
```

### Part D — Observability

```bash
# Read traces from live API pods
kubectl logs deployment/api -n rahbar | python -m json.tool | grep trace_id

# Port-forward metrics endpoint
kubectl port-forward svc/api 8000:8000 -n rahbar
curl http://localhost:8000/metrics

# Run trace tool
python scripts/trace_tool.py --list
python scripts/trace_tool.py <trace_id>
```

### Part E — Coverage Check

```bash
python scripts/coverage_check.py
```

### Part F — Kubernetes Status

```bash
# Full cluster state
kubectl get all,pvc,configmap,cronjob,job -n rahbar

# Logs of completed jobs
kubectl logs job/rahbar-indexer -n rahbar
kubectl logs job/rahbar-eval    -n rahbar
```

---

## File Map

| File | Purpose |
|------|---------|
| `configmap.yaml` | Main config (embed model, collection, top-k, etc.) |
| `gate-configmap.yaml` | Release gate thresholds (Part C2) |
| `pvcs.yaml` | PVCs for datalake, traces, eval results |
| `api.yaml` | Live API Deployment + Service |
| `api-candidate.yaml` | Candidate Deployment for Part C3 |
| `chroma.yaml` | ChromaDB StatefulSet + Service |
| `redis.yaml` | Redis cache Deployment |
| `indexer-job.yaml` | One-shot indexer Job (with verification) |
| `reindex-cronjob.yaml` | Weekly re-index CronJob |
| `verify-cronjob.yaml` | Daily verify+ingest CronJob (Part F2) |
| `eval-job.yaml` | Evaluation Job (Part F2) |
| `networkpolicy.yaml` | Only indexer writes to ChromaDB |
| `hpa.yaml` | Horizontal pod autoscaler for API |
