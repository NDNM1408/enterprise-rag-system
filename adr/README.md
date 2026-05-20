# Architecture Decision Records

Mỗi file `NNNN-<slug>.md` ghi lại một quyết định kiến trúc — bối cảnh, lựa chọn, lý do, hậu quả — để 6 tháng sau ai vào repo cũng hiểu **vì sao** code đang trông như vậy.

Format (rút gọn từ Michael Nygard's ADR template):

```
# ADR-NNNN: <tiêu đề ngắn>

- Status: Proposed | Accepted | Superseded by ADR-XXXX
- Date: YYYY-MM-DD
- Deciders: <ai đã quyết>

## Context
Vấn đề + ràng buộc tại thời điểm quyết định.

## Decision
Chốt cái gì.

## Rationale
Bằng chứng + lý lẽ dẫn tới quyết định.

## Alternatives considered
Phương án A/B/C đã cân nhắc và vì sao loại.

## Consequences
Trade-off, follow-up cần làm.
```

## Mục lục

| ID | Title | Status |
|---|---|---|
| [0001](0001-markdown-chunking-strategy.md) | Markdown chunking: section-as-paragraph với child split | Accepted (2026-05-20) |

## Tài liệu phụ trợ

- [`examples/`](examples/) — mock chunks minh hoạ output của mỗi strategy trên cùng 1 input
- [`assets/`](assets/) — biểu đồ, bảng so sánh trích từ benchmark
