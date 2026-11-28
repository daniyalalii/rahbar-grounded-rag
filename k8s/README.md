# Kubernetes Setup Guide — RAHBAR RAGOps (Assignment 03)

This guide provides **exact, tested commands** to deploy, run, and demonstrate all Kubernetes parts (F1–F4) of Assignment 03.

> **Prerequisites**: `minikube`, `kubectl`, `docker` installed and on PATH.  
> All commands are for PowerShell on Windows.

---

## 1. Start the cluster

```powershell
minikube start --driver=docker --cpus=4 --memory=6144 --addons=ingress
# Point Docker CLI to the minikube daemon so local images are visible
& minikube -p minikube docker-env --shell=powershell | Invoke-Expression
```

---

## 2. Build images

```powershell
docker build -t rahbar-api:dev    -f api/Dockerfile    .
docker build -t rahbar-indexer:dev -f indexer/Dockerfile .
# (Optional) candidate image with same code but different config via env vars
docker tag rahbar-api:dev rahbar-api:candidate
```

---

## 3. Deploy base stack (Part F1)

```powershell
# Apply the full dev overlay (sets namespace rahbar, all base resources)
kubectl apply -k k8s/overlays/dev
kubectl -n rahbar rollout status deployment/api
kubectl -n rahbar rollout status deployment/chroma
kubectl -n rahbar rollout status deployment/redis
```

Expected: three deployments `Running`.

Verify PVCs are bound:

```powershell
kubectl -n rahbar get pvc
# NAME               STATUS   CAPACITY
# datalake-pvc       Bound    2Gi
# traces-pvc         Bound    1Gi
# eval-results-pvc   Bound    500Mi
```

---

## 4. Initial index (Part F2 — indexer Job)

```powershell
kubectl apply -f k8s/base/indexer-job.yaml -n rahbar
kubectl -n rahbar wait --for=condition=complete job/indexer --timeout=300s
kubectl -n rahbar logs job/indexer
```

Look for log lines confirming:

```
newly_embedded_chunks: <N>
unchanged_chunks: 0
deleted_chunks: 0
```

Re-running the same job (unchanged corpus) must show `newly_embedded_chunks: 0`.

---

## 5. Run evaluation suite (Part F2 — eval Job)

```powershell
kubectl apply -f k8s/base/eval-job.yaml -n rahbar
kubectl -n rahbar wait --for=condition=complete job/rahbar-eval --timeout=600s
kubectl -n rahbar logs job/rahbar-eval
```

The job writes a timestamped JSON to the `eval-results-pvc` and exits with code 0 (gate pass) or 1 (gate fail).

Retrieve the result file from the pod after completion:

```powershell
# Spin up a debug pod to read the PVC
kubectl -n rahbar run pvc-reader --image=busybox --rm -it --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"v","persistentVolumeClaim":{"claimName":"eval-results-pvc"}}],"containers":[{"name":"c","image":"busybox","command":["sh"],"volumeMounts":[{"mountPath":"/data","name":"v"}]}]}}'
# Inside the pod:
ls /data/
cat /data/<timestamped>.json
```

---

## 6. Weekly reindex CronJob (Part F2)

```powershell
# CronJob is already deployed via the base overlay.
# To trigger it manually for a demo:
kubectl -n rahbar create job reindex-manual --from=cronjob/reindex
kubectl -n rahbar wait --for=condition=complete job/reindex-manual --timeout=300s
kubectl -n rahbar logs job/reindex-manual
```

The datalake PVC is mounted (`/datalake`) so hash store persists between runs; unchanged documents embed zero chunks.

---

## 7. Candidate experiment and release gate (Part F3 / Part C)

### 7a. Build the candidate collection

```powershell
kubectl apply -f k8s/base/indexer-job-candidate.yaml -n rahbar
kubectl -n rahbar wait --for=condition=complete job/indexer-candidate --timeout=300s
kubectl -n rahbar logs job/indexer-candidate
```

### 7b. Deploy candidate API

```powershell
kubectl apply -f k8s/base/api-candidate.yaml -n rahbar
kubectl -n rahbar rollout status deployment/api-candidate
```

### 7c. Run gate against candidate

```powershell
# Apply the candidate eval job (edit --api flag to point at api-candidate:8000)
kubectl -n rahbar run eval-candidate --image=rahbar-indexer:dev --rm -it --restart=Never \
  --env="TRACE_DIR=/tmp" \
  -- python /app/eval/run_eval_ragops.py \
       --api http://api-candidate:8000 \
       --golden /app/eval/golden.jsonl \
       --label candidate \
       --out-dir /tmp
```

### 7d. Gate pass → rolling update; gate fail → delete candidate

```powershell
# If gate PASSED:
kubectl -n rahbar set image deployment/api api=rahbar-api:candidate
kubectl -n rahbar rollout status deployment/api

# If gate FAILED (do NOT promote):
kubectl delete -f k8s/base/api-candidate.yaml -n rahbar

# Rollback live deployment to previous image at any time:
kubectl -n rahbar rollout undo deployment/api
kubectl -n rahbar rollout status deployment/api
```

---

## 8. Tracing and metrics (Part D)

```powershell
# Port-forward the API
kubectl -n rahbar port-forward svc/api 8000:8000

# In a second terminal — send a query and check the trace
curl -s http://localhost:8000/query -d '{"question":"What CGPA is needed for graduation?"}' | python -m json.tool

# List trace files
kubectl -n rahbar run trace-reader --image=busybox --rm -it --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"v","persistentVolumeClaim":{"claimName":"traces-pvc"}}],"containers":[{"name":"c","image":"busybox","command":["sh"],"volumeMounts":[{"mountPath":"/traces","name":"v"}]}]}}'
# Inside: ls /traces/

# Inspect a specific trace
.\.venv\Scripts\python.exe scripts/trace_tool.py <trace-id>
```

---

## 9. Query coverage and drift (Part E)

```powershell
# Before (original corpus, original golden set)
.\.venv\Scripts\python.exe scripts/coverage_check.py `
  --golden eval/golden_before.jsonl `
  --live   scripts/live_queries.jsonl `
  --sim-threshold 0.60 `
  --min-coverage  0.85 `
  --out eval/results/coverage_before.json

# After (expanded golden set including uncovered topics)
.\.venv\Scripts\python.exe scripts/coverage_check.py `
  --golden eval/golden.jsonl `
  --live   scripts/live_queries.jsonl `
  --sim-threshold 0.60 `
  --min-coverage  0.85 `
  --out eval/results/coverage_after.json
```

---

## 10. Verify all resources

```powershell
kubectl -n rahbar get all,pvc,configmap,cronjob,job
kubectl -n rahbar describe configmap rahbar-config
kubectl -n rahbar describe configmap ragops-gate-thresholds
```

---

## 11. Gate ConfigMap — change thresholds without rebuild

```powershell
# Edit thresholds live:
kubectl -n rahbar edit configmap ragops-gate-thresholds
# Or apply the file:
kubectl apply -f k8s/base/gate-configmap.yaml -n rahbar
```

Current thresholds (`gate-configmap.yaml`):

| Key | Default |
|-----|---------|
| `RECALL_AT_5_MAX_DROP` | 0.05 |
| `FAITHFULNESS_MIN` | 0.70 |
| `P95_LATENCY_MAX_MS` | 5000 |
| `ABSTENTION_CORRECTNESS_MIN` | 1.0 |
| `CITATION_VALIDITY_MIN` | 0.90 |

---

## 12. Namespace note

All manifests are deployed to the `rahbar` namespace (set in `k8s/base/kustomization.yaml`).  
Every `kubectl` command must include `-n rahbar` or use `kubectl apply -k` which inherits the namespace from Kustomize.

---

## 13. Cluster teardown

```powershell
kubectl delete namespace rahbar   # removes all resources incl. PVCs
minikube stop
minikube delete                   # optional: wipe the cluster entirely
```
