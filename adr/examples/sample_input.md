**Status**

| Owner | Pham Kim Long |
| --- | --- |
| Reviewer | Hai Ngoc Nguyen |

# I. BUSINESS CONTEXT

## 1.1 Introduction

The TCB – OMG – MAG ecosystem has ambitious targets to grow over the next 5 years.
The Keystone initiative comprises two foundational platforms: the **Federated Data
Platform (FDP)** and the **Agentic AI Platform**.

## 1.2 Scope

In-scope:

- Agent registry: register, deploy, publish
- Cross-entity discovery

# II. PROPOSED SOLUTION

## 2.2.4 Tech Stack Summary

| Component | Technology | Version |
| --- | --- | --- |
| API Gateway | Kong | 3.9.1 |
| Identity Provider | Keycloak | 26.5.7 |
| Vector DB | Qdrant | 1.17.1 |

## 2.6.3 Agent Lifecycle Flows

### SD-05: Agent Deployment Pipeline

The agent state machine transitions through:
`registered → deploying → running → published → retired → error`.

Running → error triggers:

1. Canary error rate exceeds threshold → auto-rollback
2. Heartbeat stops for > 90 seconds → marked unhealthy

When error state is entered, the deployment controller rolls back to the previous
published version and emits an `agent.lifecycle.error` event to Kafka.
