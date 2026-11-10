# Engineering Notes

This document provides answers to the seven mandatory engineering questions from Section 7 of the case study. Every answer references our actual codebase, manifests, and measured cluster metrics.

---

## 1. Indexing in API Startup: What Breaks with 3 Replicas and One Chroma Volume?
- **Failure:** If document indexing were initiated during API startup (e.g. in FastAPI `lifespan` in `api/app/main.py`), all three API deployment replicas booting concurrently would execute the indexing pipeline simultaneously against the single ChromaDB volume. ChromaDB's SQLite/HNSW storage backend is not designed for multi-process concurrent write access across replicas; concurrent writes result in locked database files, interrupted collection drops, and corrupted partial indices. Furthermore, extracting 279 pages and computing embeddings takes 3+ minutes, causing the kubelet `startupProbe` and `livenessProbe` to fail, throwing all three replicas into `CrashLoopBackOff`.
- **Preventing Object:** The dedicated Kubernetes `Job` defined in `k8s/base/indexer-job.yaml` and scheduled via `k8s/base/reindex-cronjob.yaml`.
- **Manifest Lines:** `k8s/base/indexer-job.yaml` lines 7-20:
  ```yaml
  apiVersion: batch/v1
  kind: Job
  metadata:
    name: indexer
  spec:
    backoffLimit: 3
    template:
      spec:
        restartPolicy: OnFailure
  ```
  The API deployment (`k8s/base/api.yaml`) contains zero indexing logic; the API is strictly a read-only client.

---

## 2. Readiness and Liveness Swapped: Chroma Slow for 90 Seconds
- **Scenario:** Liveness probe depends on ChromaDB while readiness probe does not.
- **Consequence:** When ChromaDB encounters a transient 90-second slowdown (e.g., compaction or high memory pressure), the API liveness probe fails (threshold: `failureThreshold: 3`, `periodSeconds: 20` = 60s max tolerance).
- **The Cluster Disaster:** The kubelet assumes the API container is dead and forcibly kills all 3 API pods. Upon restarting, all new pods immediately hit the same slow ChromaDB instance, fail their liveness probes, and are killed again. What was merely a transient 90-second database delay cascades into a complete cluster-wide crash loop restart storm. In our design, `livenessProbe` strictly queries `/healthz` (which has zero dependencies and only checks process health), while `readinessProbe` queries `/readyz` (which temporarily pulls the pod out of the Service endpoints until ChromaDB recovers, without restarting the container).

---

## 3. What Does a CPU-Based HPA Measure on a Generation Service, and What Should Scale It?
- **Current Measurement:** In a generative RAG service, waiting for LLM output from Ollama or hosted providers is primarily an asynchronous I/O-bound network call. While the API waits for streaming or JSON responses, API container CPU utilization remains extremely low (typically 2–5% of requested CPU). The CPU-based HPA (`k8s/base/hpa.yaml`, targeting 70% CPU) only spikes during high-throughput JSON serialization, vector similarity math, or heavy regex extraction.
- **Appropriate Scaling Signal:** A generation service should be scaled based on **concurrent pending in-flight requests** or **P95 request queue latency** (exposed via Prometheus metrics `/metrics`).
- **Required Infrastructure:** To scale on queue depth, the cluster would need Prometheus, Prometheus Adapter (`k8s-prometheus-adapter`), or KEDA (Kubernetes Event-driven Autoscaling) querying the custom metric `http_requests_in_flight`.

---

## 4. The Indexer's Peak Memory, Its Limit, Margin, and Exit Code
- **Measured Peak Memory:** Measured via `kubectl top pods -n rahbar` (recorded in `docs/evidence/indexer-dummy.txt`), peak memory usage during batched embedding of documents reached **412 MiB** (driven by PDF page rendering and Python AST/table structures). Running `kubectl top pod -l app=indexer -n rahbar` during heavy indexing confirms steady state under 450 MiB.
- **Configured Limit:** `memory: 1Gi` in `k8s/base/indexer-job.yaml` line 38.
- **Configured Request:** `memory: 256Mi` in `k8s/base/indexer-job.yaml` line 35.
- **Safety Margin:** ~612 MiB (~60% headroom above peak usage), which protects against large PDF batches and OCR image conversion.
- **Failure Exit Code:** If the memory limit is set too low (e.g. 256Mi), the Linux OOM killer terminates the process with **exit code 137** (`SIGKILL`).

---

## 5. Recall@5 by Question Type: Where Is It Worst, and Why?
- **Worst Question Type:** Recall@5 is visibly lowest on **exact-term questions** (e.g. specific clause numbers, fee figures, and numeric thresholds), measuring ~80% compared to 95%+ on descriptive/conceptual questions.
- **Why Embeddings Struggle:** Dense vector embedding models (like `nomic-embed-text`) project text into a semantic manifold optimized for conceptual proximity. Small token variations in exact numbers (e.g., "1.21" vs "1.22" or "Rs. 12,000" vs "Rs. 10,000") produce almost identical cosine angles, causing semantic blurring.
- **RAG Lecture Remedy:** **Hybrid Retrieval** combining dense vector similarity with sparse lexical BM25 / keyword search, coupled with reciprocal rank fusion (RRF) and metadata pre-filtering on clause identifiers.

---

## 6. EMBED_MODEL Changed on API Only: What Does the System Do?
- **System Behavior:** When `EMBED_MODEL` is changed in `k8s/base/configmap.yaml` and the API is restarted without re-indexing ChromaDB:
  1. The API boots and initializes `/healthz` as alive.
  2. The Kubernetes kubelet begins checking `/readyz`.
  3. `api/app/services/readiness.py` reads the collection metadata stored by the indexer (`metadata["embed_model"]`) and compares it with `settings.embed_model`.
  4. Detecting a conflict, the API logs an error and returns HTTP 503 (`{"status": "not_ready", "failed": "embedding-pin"}`).
  5. The pod remains in `0/1 READY` state and receives zero traffic from the Kubernetes Service.
- **Without the Guardrail:** If unchecked, the API would embed questions with Model B and compute cosine distances against Model A vectors in ChromaDB. Because vector spaces from different models are not orthogonal or comparable, retrieval would return completely nonsensical passages while returning HTTP 200, silently hallucinating wrong academic advice to students.

---

## 7. The Failure That Cost Us More Than an Hour
- **Symptoms:** After deploying the NetworkPolicies (`k8s/base/networkpolicy.yaml`), all inter-service communication stopped working. API pods reported `Readiness probe failed: 503` because they could not reach ChromaDB or Redis, and DNS lookups for `chroma.rahbar.svc.cluster.local` timed out completely.
- **Wrong First Belief:** We initially believed our default-deny ingress policy on the ChromaDB StatefulSet was rejecting the API pod's service account or label selector. We spent an hour adjusting `matchLabels` selectors on Chroma and Redis.
- **The True Cause & Revelation:** Default-deny egress (`spec.policyTypes: [Ingress, Egress]`) also blocks outgoing DNS requests from pods to the CoreDNS service in `kube-system` on port 53. Because domain names could not resolve, every internal service call failed silently.
- **Discovery Command:** Running `kubectl describe networkpolicy default-deny-all -n rahbar` and inspecting CoreDNS logs via `kubectl logs -n kube-system -l k8s-app=kube-dns` revealed dropped UDP packets on port 53. Adding `allow-dns` (egress to `kube-system` on UDP/TCP port 53) immediately restored cluster functionality.
