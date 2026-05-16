# ADR-002: pgvector + PostgreSQL as the Vector Store

**Status:** Accepted  
**Date:** 2024-Q4  

---

## Context

The RAG pipeline requires:
- Dense vector similarity search (semantic retrieval)
- Full-text search (BM25-style keyword matching for document names, well IDs)
- Hybrid retrieval with Reciprocal Rank Fusion (RRF)
- Metadata filtering (filter by field_name, document_type)
- ACID transactions (citations and investigations in same DB as vectors)
- Deployment in a regulated offshore environment (no vendor SaaS)

---

## Decision

Use **pgvector** extension on **PostgreSQL 16** as the unified vector and relational store.

---

## Options Considered

| Store | Verdict | Reasons |
|-------|---------|---------|
| **pgvector + PostgreSQL** | ✅ CHOSEN | Hybrid vector + FTS in one engine. ACID. Self-hosted. Familiar ops toolchain. No separate service dependency. RRF fusion in SQL. Azure Database for PostgreSQL supports pgvector natively. |
| **Chroma** | ❌ Rejected | Embedded or single-server mode only. No SQL joins. ACID guarantees unclear. Adding FTS requires BM25 index in a separate service. |
| **Pinecone** | ❌ Rejected | SaaS-only. Data sovereignty concern for Norwegian energy sector (GDPR + PSA data locality). No on-prem option. Separate service adds network latency and failure mode. |
| **Weaviate** | ❌ Rejected | Self-hostable, supports hybrid search. But introduces a separate stateful service, requiring separate backup, HA, and ops runbooks — increased operational complexity vs unified PostgreSQL. |
| **Qdrant** | ❌ Rejected | Good performance. But again a separate service. No SQL-level metadata joins without additional plumbing. |
| **Milvus** | ❌ Rejected | Kubernetes-native, operationally complex for a 1-service deployment. Overkill for <100k document chunks. |

---

## Implementation

```sql
-- Hybrid retrieval: semantic + FTS + RRF fusion
WITH semantic AS (
  SELECT id, content, 1 - (embedding <=> $query_vec) AS score
  FROM document_chunks
  ORDER BY embedding <=> $query_vec
  LIMIT 20
),
keyword AS (
  SELECT id, content, ts_rank(fts_vector, query) AS score
  FROM document_chunks, plainto_tsquery('english', $query_text) query
  WHERE fts_vector @@ query
  LIMIT 20
)
-- RRF: 1/(k + rank) where k=60
SELECT id, content FROM (
  SELECT id, content, SUM(1.0 / (60 + rank)) AS rrf_score
  FROM (
    SELECT id, content, ROW_NUMBER() OVER (ORDER BY score DESC) AS rank FROM semantic
    UNION ALL
    SELECT id, content, ROW_NUMBER() OVER (ORDER BY score DESC) AS rank FROM keyword
  ) ranked
  GROUP BY id, content
) fused ORDER BY rrf_score DESC LIMIT $top_k;
```

## Consequences

### Positive
- Single PostgreSQL instance serves relational data, vectors, and FTS — one backup policy, one connection pool, one ops runbook
- `langchain-postgres` provides pgvector integration with minimal boilerplate
- Azure Database for PostgreSQL Flexible Server supports pgvector natively — Terraform `azurerm_postgresql_flexible_server_configuration` with `shared_preload_libraries = pgvector`
- ACID transactions mean embedding + metadata insert is atomic — no partial state possible
- RRF fusion is a single SQL query — inspectable, debuggable, no black-box service calls

### Negative
- pgvector HNSW index requires `CREATE INDEX` on first deploy (handled by `init.sql`)
- Not a purpose-built ANN engine — at >1M chunks, Qdrant/Weaviate would be faster
- For the ~50k-100k chunk scale of this project, pgvector IVFFLAT or HNSW is fast enough (<50ms p95)
