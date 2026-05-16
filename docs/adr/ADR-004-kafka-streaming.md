# ADR-004: Kafka for Anomaly Event Streaming

**Status:** Accepted  
**Date:** 2024-Q4  

---

## Context

Offshore SCADA systems produce continuous high-frequency telemetry (1-second tag updates, thousands of tags per platform). The anomaly detection pipeline must:
- Consume telemetry in near-real-time
- Decouple the producer (SCADA/historian integration) from the consumer (anomaly detector)
- Allow multiple consumers (anomaly detector, data archiver, dashboard feed)
- Provide replay capability for post-incident forensics
- Handle connectivity interruptions (offshore platforms have intermittent uplinks)

In production (Azure), this maps to **Azure Event Hubs** (Kafka protocol compatible).  
In development, this uses **Confluent Kafka** via Docker Compose.

---

## Decision

Use **Apache Kafka / Azure Event Hubs** (Kafka protocol) for event streaming.  
Topics: `well.telemetry`, `well.anomalies`, `agent.escalations`.

---

## Options Considered

| Option | Verdict | Reasons |
|--------|---------|---------|
| **Kafka / Azure Event Hubs** | ✅ CHOSEN | Industry standard for high-throughput telemetry. Replay capability. Consumer group semantics. Event Hubs uses the same Kafka protocol — no code change between dev and prod. Supported by Aker BP's CDF/Fabric platform. |
| **RabbitMQ** | ❌ Rejected | Message queue, not an event log. No replay/rewind. Not suitable for telemetry time-series that need forensic replay. |
| **Azure Service Bus** | ❌ Rejected | Queue semantics (at-most-once or at-least-once delivery). Not designed for high-throughput telemetry. No partition-level parallelism. Better suited for workflow tasks (used for escalation notifications, not telemetry ingestion). |
| **PostgreSQL LISTEN/NOTIFY** | ❌ Rejected | Suitable for low-volume events. Not appropriate for thousands of telemetry tags/second. No replay. |
| **Redis Streams** | ❌ Rejected | Good for low-to-medium throughput. Not supported natively by Azure Event Hubs Kafka endpoint. Adds another stateful service. |

---

## Topic Design

| Topic | Producer | Consumers | Retention | Notes |
|-------|----------|-----------|-----------|-------|
| `well.telemetry` | SCADA integration / synthetic generator | Anomaly detector | 7 days | Raw telemetry; partition by well_id |
| `well.anomalies` | Anomaly detector | Investigation API, dashboard | 30 days | Structured anomaly events with scores |
| `agent.escalations` | Investigation API | Operations center, NOC | 90 days | Escalation records; compliance retention |

---

## Dev/Prod Parity

```python
# Identical client code for Confluent (dev) and Azure Event Hubs (prod)
# Only the bootstrap server and SASL config differs
config = {
    "bootstrap.servers": settings.kafka_bootstrap_servers,
    # Dev: no auth; Prod: Event Hubs SASL/PLAIN with connection string
}
```

The Terraform configuration provisions Azure Event Hubs with Kafka endpoint enabled:
```hcl
resource "azurerm_eventhub_namespace" "northsea" {
  kafka_enabled = true
  # Kafka endpoint available at: {namespace}.servicebus.windows.net:9093
}
```

**Gap to close**: `src/config.py` needs `kafka_sasl_username` and `kafka_sasl_password` (or Azure connection string) for the production auth configuration. Currently only dev (no-auth) config is present.

---

## Consequences

- Replay capability: anomaly detector can be restarted and reprocess the last 7 days of telemetry
- Consumer group isolation: adding a new consumer (e.g. data archiver) does not impact existing consumers
- Azure Event Hubs upgrade path: change only the bootstrap server and add SASL config — no code change
- Dev/prod parity via Confluent image in Docker Compose and Event Hubs Kafka endpoint in Azure
