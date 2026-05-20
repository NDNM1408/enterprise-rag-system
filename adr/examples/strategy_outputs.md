# Strategy outputs trên cùng input

Input: [`sample_input.md`](sample_input.md) — trích lược DAB1 (Status preamble + 4 heading sections + 1 sub-section sâu).

Mỗi strategy được mô tả qua **một cặp** `content` (cosine vào query) + `parent_text` (LLM thấy khi hit). Câu hỏi truy vấn giả định: *"What's the state machine for an agent?"*

---

## 1. `sentence` — LlamaIndex SentenceSplitter (chunk_size=512, overlap=50)

Sliding window theo token, không biết cấu trúc markdown.

**Chunk #4 (rank-1 cho query):**

```text
content == parent_text:
"In-scope: - Agent registry: register, deploy, publish - Cross-entity discovery

# II. PROPOSED SOLUTION

## 2.2.4 Tech Stack Summary

| Component | Technology | Version |
| --- | --- | --- |
| API Gateway | Kong | 3.9.1 |
...
## 2.6.3 Agent Lifecycle Flows

### SD-05: Agent Deployment Pipeline

The agent state machine transitions through:
`registered → deploying → running → published → retired → error`."
```

**Vấn đề:** Cắt giữa 2 section khác nhau (Tech Stack table + SD-05). Vector chứa nhiều topic → cosine "loãng". Câu trả lời (state machine list) bị nhồi cùng table Kong/Keycloak/Qdrant — embedding bị phân tán.

---

## 2. `hierarchical` — LlamaIndex HierarchicalNodeParser(chunk_sizes=[1024, 256])

Cắt sliding window ở 2 resolution. Embed child 256-tok, payload mang parent 1024-tok.

**Chunk #7 (rank-1 cho query):**

```text
content (child, ~256 tok):
"The agent state machine transitions through:
registered → deploying → running → published → retired → error.

Running → error triggers:
1. Canary error rate exceeds threshold → auto-rollback
2. Heartbeat stops for > 90 seconds"

parent_text (parent, ~1024 tok):
"<chứa cả content trên + window xung quanh, bao gồm tail của Tech Stack table và
header SD-05, nhưng không có heading metadata>"
```

**Ưu:** Child nhỏ → vector precise. Parent rộng → LLM context đủ.
**Nhược:** Không hiểu cấu trúc markdown. Header path không metadata, parent có thể cắt giữa table.

---

## 3. `parent_child` (max=2048, "section thuần") — **MarkdownSplitter của repo**

1 leaf section = 1 chunk khi vừa. Section quá to mới chia.

**Chunk #5 (rank-1 cho query):**

```text
heading_path: "II. PROPOSED SOLUTION > 2.6.3 Agent Lifecycle Flows > SD-05: Agent Deployment Pipeline"

content (== parent_text khi section vừa):
"### SD-05: Agent Deployment Pipeline

The agent state machine transitions through:
registered → deploying → running → published → retired → error.

Running → error triggers:
1. Canary error rate exceeds threshold → auto-rollback
2. Heartbeat stops for > 90 seconds → marked unhealthy

When error state is entered, the deployment controller rolls back to the previous
published version and emits an agent.lifecycle.error event to Kafka."
```

**Ưu:** Section nguyên khối + heading_path metadata. Vector clean (1 topic).
**Nhược:** Khi section RẤT to (vd Tech Stack với 30 rows), chunk thành 3000+ token, vector dilute.

---

## 4. `pc_med_deepest` (max=512, prefix=deepest) — **PRODUCTION DEFAULT**

Section ≤512 tok = 1 chunk. Section to → chia con, parent vẫn giữ nguyên section đầy đủ.
Embed prefix chỉ chứa LEAF heading (không full path).

**Chunk #8 (rank-1 cho query):**

```text
heading_path: "II. PROPOSED SOLUTION > 2.6.3 Agent Lifecycle Flows > SD-05: Agent Deployment Pipeline"

content (~256 tok embed string — chỉ deepest heading):
"### SD-05: Agent Deployment Pipeline

The agent state machine transitions through:
registered → deploying → running → published → retired → error.

Running → error triggers:
1. Canary error rate exceeds threshold → auto-rollback
2. Heartbeat stops for > 90 seconds → marked unhealthy"

parent_text (~512 tok full leaf section với FULL heading path):
"# II. PROPOSED SOLUTION
## 2.6.3 Agent Lifecycle Flows
### SD-05: Agent Deployment Pipeline

The agent state machine transitions through:
registered → deploying → running → published → retired → error.
...
When error state is entered, the deployment controller rolls back to the previous
published version and emits an agent.lifecycle.error event to Kafka."
```

**Ưu:** Section vừa thì content==parent. Section to thì child precise + parent đầy đủ context. Deepest-heading-only giảm boilerplate trùng giữa các chunks cùng cha.
**Nhược:** Section khổng lồ (>512 tok) sẽ split — không "1 section = 1 chunk" thuần.

---

## 5. `pc_small_deepest` (max=256, prefix=deepest) — **WINNER trên benchmark**

Như `pc_med_deepest` nhưng child nhỏ hơn → vector precise hơn.

**Chunk #12 (rank-1 cho query):**

```text
heading_path: "II. PROPOSED SOLUTION > 2.6.3 Agent Lifecycle Flows > SD-05: Agent Deployment Pipeline"

content (~150 tok embed string):
"### SD-05: Agent Deployment Pipeline

The agent state machine transitions through:
registered → deploying → running → published → retired → error.

Running → error triggers:
1. Canary error rate exceeds threshold → auto-rollback
2. Heartbeat stops for > 90 seconds → marked unhealthy"

parent_text (~512 tok full leaf section như pc_med_deepest):
"# II. PROPOSED SOLUTION
## 2.6.3 Agent Lifecycle Flows
### SD-05: Agent Deployment Pipeline
...
When error state is entered, the deployment controller rolls back...
emits an agent.lifecycle.error event to Kafka."
```

**Ưu:** Child cô đọng nhất → top-1 hit rate cao nhất trên benchmark (16/20).
**Nhược:** Số chunks gấp 2× (220 vs 109 cho doc DAB1).

---

## So sánh trực quan trên câu hỏi: "Who owns this document?"

Câu trả lời nằm ở **preamble** (`**Status**` table, trước heading đầu).

| Strategy | Chunk chứa câu trả lời ở rank | Lý do |
|---|---:|---|
| sentence | 1 | Sliding window từ đầu doc — catch preamble tự nhiên |
| hierarchical | 1 | Cùng lý do |
| parent_child | 3 | Preamble chunk không có heading anchor → cosine thấp hơn chunks có heading "1.1 Introduction" |
| pc_med_deepest | 4 | Như trên + smaller chunks → nhiều heading-anchored chunks chèn lên trước |
| pc_small_deepest | 4 | Như trên |

→ **Trade-off được ghi nhận trong ADR-0001:** PC variants tốt cho retrieval có cấu trúc nhưng yếu hơn cho preamble/title-page content. Mitigation: preamble-fix trong `_iter_leaf_sections` đảm bảo preamble vẫn được retrieve, chỉ ở rank thấp hơn — vẫn nằm trong top-5.
