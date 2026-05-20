# Benchmark scorecard — snapshot 2026-05-20

Snapshot từ manual judge (đọc 20 câu × 5 strategy × 5 chunk = 500 parent_texts). Source data: `benchmark/results/judgments.json`.

## Aggregate

| strategy | answerable | rank=1 hit | rank≤3 hit | avg rank (penalty=6 nếu cần combine) |
|---|---:|---:|---:|---:|
| `sentence` (LlamaIndex 512/50) | 18/20 | 9/20 | 13/20 | 2.85 |
| `hierarchical` (LlamaIndex 1024/256) | 20/20 | 15/20 | 18/20 | 1.75 |
| `parent_child` (max=2048, full prefix) | 19/20 | 14/20 | 18/20 | 1.75 |
| **`pc_med_deepest` (max=512, deepest)** ← prod default | **20/20** | **13/20** | **18/20** | **1.75** |
| `pc_small_deepest` (max=256, deepest) ← benchmark winner | 20/20 | 16/20 | 18/20 | 1.50 |

## Câu khó (per strategy)

| strategy | weak questions |
|---|---|
| `sentence` | Q7, Q10, Q12, Q16, Q18 |
| `hierarchical` | Q10, Q16 |
| `parent_child` | Q7, Q16 |
| `pc_med_deepest` | Q16 |
| `pc_small_deepest` | Q16 |

→ **Q16 (TCB→OMG cross-entity chain)**: cả 5 strategy đều phải combine 2+ chunks. Câu hỏi vượt ngoài retrieval thuần — cần multi-hop agent.

## Chunk statistics on DAB1 (1551 dòng, ~12K tokens)

| strategy | # chunks | avg chars | median | min/max |
|---|---:|---:|---:|---:|
| sentence | 80 | 1757 | 1814 | 417/2478 |
| hierarchical | 167 | 812 | 874 | 54/1258 |
| parent_child (max=2048) | 56 | 2416 | 1947 | 151/7521 |
| pc_med_deepest (max=512) | 109 | 1264 | 1446 | 111/2378 |
| pc_small_deepest (max=256) | 220 | 683 | 708 | 60/1660 |

→ `pc_med_deepest` cân bằng tốt nhất: chunks vừa phải, retrieval ngang hierarchical.
