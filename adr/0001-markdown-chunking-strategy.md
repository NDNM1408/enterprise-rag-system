# ADR-0001: Markdown chunking — section-as-paragraph với child split

- **Status:** Accepted
- **Date:** 2026-05-20
- **Deciders:** MINHNGUYEN38

## Context

Pipeline ingest của `data-processing-job` cắt markdown thành chunks rồi embed lên Qdrant cho RAG. Quyết định chiến lược chunking ảnh hưởng trực tiếp:

1. **Retrieval precision** — vector quá to → mỗi vector trộn nhiều topic → cosine kém.
2. **Generation quality** — LLM cần context xung quanh hit để trả lời đúng → child cô lập không đủ.
3. **Storage + cost** — số chunks tăng tuyến tính với độ chia nhỏ.

Trước ADR này, code chạy `MarkdownSplitter` mặc định `retrieve_max_tokens=2048` / `retrieve_target_tokens=1800` với heading-path prefix đầy đủ trong content. Quan sát 2 vấn đề:

- **Preamble loss**: text trước heading đầu tiên (status table, title page) bị `_iter_leaf_sections` skip im lặng.
- **Greedy-pack rời rạc**: 1 leaf section bị xé thành N chunks, mỗi chunk lại trộn vài paragraph khác topic — vector dilute.

Cần xác định **đơn vị chunking tự nhiên** và **chiến lược tách khi quá to**, có evidence từ benchmark.

## Decision

`MarkdownSplitter` áp dụng mô hình **section-as-paragraph**:

1. **Đơn vị tự nhiên = leaf section dưới heading**. Bullets / lists / tables thuộc section đó.
2. **Tables/images là metadata của section** (không tách thành chunk riêng).
3. **Một section vừa `retrieve_max_tokens` → 1 chunk**. `content == parent_text`.
4. **Section quá to → chia children theo paragraph/table blocks**. Mọi child share `parent_text = full section`.
5. **`parent_text` luôn carry full heading trail**; `content` chỉ carry deepest heading (`content_prefix_mode="deepest"`).
6. **Preamble (text trước `#` đầu tiên) được emit thành chunk độc lập** với `heading_path=None`.

**Production default** (sau benchmark):

```python
RETRIEVE_MAX_TOKENS = 512
RETRIEVE_TARGET_TOKENS = 400      # informational, unused by new logic
CONTENT_PREFIX_MODE = "deepest"
```

## Rationale

Đối chiếu 5 strategies trên dataset DAB1 (1551 dòng markdown) + 20 câu test, **manual judge** (đọc từng top-5 parent_text):

| strategy | answerable | rank=1 hit | rank≤3 | avg rank |
|---|---:|---:|---:|---:|
| `sentence` (LlamaIndex 512/50) | 18/20 | 9/20 | 13/20 | 2.85 |
| `hierarchical` (LlamaIndex 1024/256) | 20/20 | 15/20 | 18/20 | 1.75 |
| `parent_child` (max=2048, full prefix) | 19/20 | 14/20 | 18/20 | 1.75 |
| `pc_med_deepest` (max=512, deepest) ← **prod default** | 20/20 | 13/20 | 18/20 | 1.75 |
| `pc_small_deepest` (max=256, deepest) ← benchmark winner | 20/20 | 16/20 | 18/20 | 1.50 |

Embedding: Gemini `gemini-embedding-001` 1536d (yêu cầu ban đầu là Bedrock Cohere v3 nhưng IAM marketplace block).

**Vì sao chọn `max=512 + deepest` làm default:**

1. **Answerable 20/20** — LLM có thể trả lời mọi câu hỏi với union top-5 parent_text.
2. **Tied avg rank với hierarchical** — bằng baseline mà KHÔNG mất heading awareness, table integrity.
3. **Chunks vừa phải (109 vs 220 của pc_small_deepest)** — storage + embed cost thấp hơn 2×.
4. **Cohere v3 cap 512 tokens** — nếu fix IAM xong, default này map 1-1 với embedding model.

**Vì sao "deepest heading only" thắng "full heading path":**

- Doc DAB1 nesting 3-4 level → full path tốn 25-40 tokens boilerplate/chunk.
- Mọi chunk dưới `# I. BUSINESS CONTEXT` đều bị shift vector theo cùng boilerplate → cosine không phân biệt được.
- Deepest heading vẫn giữ structural anchor (LEAF cho biết section nào) nhưng eliminate cross-section noise.
- Benchmark: full=71.5% coverage, deepest=73.8%, none=72.3%. Deepest win.

**Vì sao section-as-paragraph thắng greedy-pack:**

- Q10 (agent state machine): section SD-05 nguyên khối → rank-1 hit. Greedy-pack chia state list và trigger conditions thành 2 chunks khác → LLM phải combine, top1_all giảm.
- Q9, Q13, Q14, Q19, Q20: tương tự — câu hỏi multi-fact trong 1 section thắng dễ khi section nguyên khối.

**Trade-off đã chấp nhận:**

- **Q1 (preamble owner) rank 3-4 thay vì 1**: vì preamble chunk không có heading anchor, các chunk có heading "1.1 Introduction" cosine cao hơn cho query "Who is the owner". Mitigation: preamble vẫn nằm trong top-5 (đảm bảo bởi `_iter_leaf_sections` fix). Acceptable.
- **Section > 512 token bị chia**: không "1 section = 1 chunk" thuần. Nhưng children share `parent_text` = full section → LLM context không mất.

Xem [`examples/strategy_outputs.md`](examples/strategy_outputs.md) để thấy cụ thể `content` / `parent_text` từng strategy trả về cho 1 query mẫu.

## Alternatives considered

### A) LlamaIndex `HierarchicalNodeParser` (chunk_sizes=[1024, 256])

- ✅ Avg rank 1.75 — ngang `pc_med_deepest`.
- ❌ Không hiểu cấu trúc markdown: cắt giữa table được, không stamp heading metadata.
- ❌ Q10 state machine vẫn cần combine (sliding window cắt giữa list + trigger).
- ❌ Phụ thuộc LlamaIndex (đã loại khỏi repo này theo MIGRATION_SUMMARY.md).

→ **Loại** vì repo đã decoupled khỏi LlamaIndex; reintroducing chỉ để chunking không xứng cost.

### B) `parent_child` với `max=2048` (section thuần, không chia child)

- ✅ Section nguyên khối cho LLM context tối đa.
- ❌ Section khổng lồ (2.7.1 Message Spec ~11KB) thành 1 chunk → vector dilute.
- ❌ **Bug Q7**: section 2.7 intro (chứa "14 topics") bị overshadow bởi 2.7.1 sub-chunks vẫn được tạo ra → miss câu hỏi.
- ❌ Answerable chỉ 19/20.

→ **Loại** vì retrieval precision yếu hơn.

### C) `pc_small_deepest` (max=256)

- ✅ Avg rank 1.50 — winner trên benchmark.
- ✅ Answerable 20/20, rank-1 hit 16/20 (cao nhất).
- ❌ Số chunks 2× (220 vs 109 cho DAB1 doc).
- ❌ Storage + embed cost 2×.
- ❌ Trên embedding model có max 512 (Cohere v3), không tận dụng được capacity.

→ **Loại làm default**, nhưng giữ làm option cho embedding model lớn (Titan v2 8192, Gemini 2048).

### D) RAGFlow naive (greedy sentence-merge, 256-token windows)

- ❌ Không heading-aware.
- ❌ Answerable 7/20 trên benchmark.
- ❌ Cắt giữa table.

→ **Loại** rõ ràng.

### E) Sentence-window (LlamaIndex SentenceWindowNodeParser, window=3)

- ✅ Embed precise (1 câu).
- ❌ 832 chunks trên 1 doc 1551 dòng — quá nhiều.
- ❌ Q10/Q12 yếu vì window quá ngắn.

→ **Loại** vì cost không cân đối với precision gain.

## Consequences

### Positive

- **Retrieval quality cao**: answerable 20/20 trên test set.
- **Decoupled khỏi LlamaIndex**: thuần Python + tiktoken.
- **Heading metadata phong phú**: `heading_path` cho header boost / breadcrumb / filtering downstream.
- **Table integrity**: không cắt giữa table; row-split chỉ khi table > max_tokens.
- **Preamble retrievable**: title pages / status blocks không bị drop.

### Negative

- **Breaking change cho prod data hiện có**: chunks generated trước ADR này dùng `max=2048, prefix=full` — cần re-ingest toàn bộ docs.
- **Số chunks tăng so với mặc định cũ**: DAB1 121 → 109 (giảm nhẹ với max=512), nhưng compared with original 56 chunks ở max=2048 thì tăng ~2×.
- **Q1 preamble retrieval ở rank 3-4 thay vì 1**: trade-off đã ghi nhận.

### Follow-up needed

- [ ] Re-ingest existing knowledge bases với config mới (script + dry-run).
- [ ] Fix IAM `aws-marketplace:Subscribe` → switch sang Bedrock Cohere v3 (chunk size matches exactly 512).
- [ ] Add header-boost retrieval (LlamaIndex pattern) tận dụng `heading_path` metadata.
- [ ] Khi nâng cấp lên embedding model > 2048 tokens, cân nhắc bumps max_tokens (giảm chunk count).

## Bằng chứng

- Benchmark setup + results: `benchmark/` (gitignored, local-only)
- Manual judge: `benchmark/results/judgments.json` — 100 verdicts (20 câu × 5 strategies)
- Scorecard aggregate: `benchmark/results/scorecard.json`
- Source code: `data-processing-job/src/app/application/core/markdown_splitter.py`
- Unit tests: `data-processing-job/tests/unit/test_markdown_splitter.py` (18 tests, all pass)
