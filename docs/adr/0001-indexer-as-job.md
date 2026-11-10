# ADR 0001: Indexer as a Kubernetes Job Rather Than API Startup

## Status
Accepted · November 2026

## Context
In a Retrieval-Augmented Generation (RAG) architecture deployed on Kubernetes, document ingestion extracts text, cleans artifacts, chunks passages, computes embedding vectors, and writes them into a persistent vector store (ChromaDB).
An initial naive design is to run indexing during the API container startup or lifespan event. However, in our production cluster architecture, the API runs with at least 3 replicas behind a HorizontalPodAutoscaler (scaling up to 8 replicas) sharing a single ChromaDB PersistentVolume.

If indexing runs on API startup:
1. **Concurrent Write Race Condition:** Three API pods booting simultaneously race to write to ChromaDB across the network. Because Chroma collection creation and insertion are not designed for concurrent multi-writer coordination on a single collection volume, concurrent writers leave a corrupted, partial index.
2. **Cascading Startup Latency & Probe Failure:** Processing 40+ pages of PDFs, OCRing scanned documents, and embedding hundreds of chunks takes several minutes. The kubelet's startup and liveness probes would timeout or exhaust failure thresholds, killing the pod in a `CrashLoopBackOff`.
3. **Redundant Re-indexing on Scale & Restart:** Every horizontal scaling event (HPA scaling 3 -> 4 replicas) or transient pod restart would trigger the full re-indexing pipeline, placing massive CPU and GPU load on the embedding model and rendering the API unavailable.
4. **Coupled Deployment:** It becomes impossible to update documentation or re-index without redeploying the entire API service, violating the principle of independent deployment cycles.

## Decision
We decouple ingestion completely from serving by packaging the indexing pipeline into a dedicated, ephemeral batch workload:
- Initial indexing runs as a Kubernetes `Job` (`k8s/base/indexer-job.yaml`) with `restartPolicy: OnFailure` and `backoffLimit: 3`.
- Periodic ingestion runs as a weekly Kubernetes `CronJob` (`k8s/base/reindex-cronjob.yaml`).
- The API is strictly a reader: it connects to ChromaDB via HTTP, reads the index, and never writes.
- The API's `/readyz` probe verifies that ChromaDB is reachable, the collection exists, is non-empty, and its embedding model metadata matches the API's configured model pin.

## Options Considered
1. **Indexing in FastAPI Lifespan / Startup:** Rejected due to write races across 3+ replicas, startup probe timeouts, and repeated indexing on pod restarts.
2. **InitContainer on the API Pod:** Rejected because all 3 API replicas in the Deployment would still each run the InitContainer simultaneously on rollout, causing the same write conflicts.
3. **Dedicated Ephemeral Job / CronJob (Chosen):** Exactly one container runs to completion, writes the collection atomically, logs output, and terminates cleanly.

## Consequences
- **Positive:** Clean separation of concerns (one writer, multiple readers). Zero risk of concurrent write corruption. Predictable resource consumption with dedicated memory limits (1Gi) isolated from the API serving limits. Re-indexing can be triggered on demand via `kubectl create job --from=cronjob/reindex manual-1 -n rahbar` with zero API downtime.
- **Negative / Tradeoffs:** Requires managing separate Kubernetes manifests (`indexer-job.yaml` and `reindex-cronjob.yaml`) and separate container images.

