-- NorthSea AgentOps database initialisation
-- Runs automatically on first container start via docker-entrypoint-initdb.d

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ─── Telemetry ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS well_telemetry (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    well_id         TEXT NOT NULL,
    field_name      TEXT NOT NULL,
    oil_rate_bopd   FLOAT NOT NULL,
    water_cut_pct   FLOAT NOT NULL,
    gas_oil_ratio   FLOAT NOT NULL,
    bhp_psi         FLOAT NOT NULL,
    wh_temp_f       FLOAT NOT NULL,
    choke_64ths     FLOAT NOT NULL,
    is_injector     BOOLEAN DEFAULT FALSE,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_telemetry_well_ts ON well_telemetry (well_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_field ON well_telemetry (field_name);

-- ─── Anomaly Alerts ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS anomaly_alerts (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    alert_id            UUID UNIQUE NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL,
    well_id             TEXT NOT NULL,
    field_name          TEXT NOT NULL,
    severity            TEXT NOT NULL CHECK (severity IN ('LOW','MEDIUM','HIGH','CRITICAL')),
    anomaly_score       FLOAT NOT NULL CHECK (anomaly_score BETWEEN 0 AND 1),
    affected_features   TEXT[] NOT NULL,
    baseline_values     JSONB NOT NULL DEFAULT '{}',
    current_values      JSONB NOT NULL DEFAULT '{}',
    deviation_pct       JSONB NOT NULL DEFAULT '{}',
    description         TEXT NOT NULL,
    investigated        BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_well ON anomaly_alerts (well_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON anomaly_alerts (severity, investigated);

-- ─── RAG Document Store ──────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    document_id     TEXT UNIQUE NOT NULL,
    title           TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    well_id         TEXT,
    field_name      TEXT,
    source_path     TEXT NOT NULL,
    total_chunks    INT DEFAULT 0,
    indexed_at      TIMESTAMPTZ DEFAULT NOW(),
    metadata        JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index     INT NOT NULL,
    content         TEXT NOT NULL,
    section         TEXT,
    page_number     INT,
    embedding       vector(1536),
    token_count     INT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Full-text search index for BM25 hybrid retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON document_chunks USING gin (to_tsvector('english', content));

-- ─── Investigations ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS investigations (
    id                      UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    investigation_id        UUID UNIQUE NOT NULL,
    alert_id                UUID REFERENCES anomaly_alerts(alert_id),
    well_id                 TEXT NOT NULL,
    timestamp               TIMESTAMPTZ NOT NULL,
    root_cause_hypothesis   TEXT NOT NULL,
    supporting_evidence     TEXT[] NOT NULL,
    recommended_actions     TEXT[] NOT NULL,
    citations               JSONB NOT NULL DEFAULT '[]',
    confidence_score        FLOAT NOT NULL,
    evidence_coverage       FLOAT NOT NULL,
    risk_level              TEXT NOT NULL,
    should_escalate         BOOLEAN NOT NULL,
    escalation_reasons      TEXT[] NOT NULL DEFAULT '{}',
    escalation_message      TEXT,
    agent_steps             JSONB NOT NULL DEFAULT '[]',
    total_tokens_used       INT DEFAULT 0,
    latency_ms              FLOAT DEFAULT 0,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_investigations_well ON investigations (well_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_investigations_escalate ON investigations (should_escalate, created_at DESC);

-- ─── Escalation Queue ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS escalations (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    escalation_id       UUID UNIQUE NOT NULL,
    investigation_id    UUID REFERENCES investigations(investigation_id),
    well_id             TEXT NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    resolved_at         TIMESTAMPTZ,
    resolved_by         TEXT,
    resolution_notes    TEXT,
    reasons             TEXT[] NOT NULL,
    risk_level          TEXT NOT NULL,
    confidence_score    FLOAT NOT NULL,
    summary             TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'PENDING'
                            CHECK (status IN ('PENDING','IN_REVIEW','RESOLVED','DISMISSED'))
);

CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations (status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_escalations_well ON escalations (well_id, status);

-- ─── Audit Log ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_audit_log (
    id                  UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    investigation_id    UUID,
    agent_id            TEXT NOT NULL,
    step_name           TEXT NOT NULL,
    tool_called         TEXT,
    input_hash          TEXT,
    output_hash         TEXT,
    tokens_used         INT DEFAULT 0,
    latency_ms          FLOAT DEFAULT 0,
    success             BOOLEAN NOT NULL,
    error_message       TEXT,
    timestamp           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_investigation ON agent_audit_log (investigation_id);
CREATE INDEX IF NOT EXISTS idx_audit_agent ON agent_audit_log (agent_id, timestamp DESC);

-- ─── MLflow tables created by mlflow itself ──────────────────────────────────
