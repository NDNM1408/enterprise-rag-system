"""Service configuration. All knobs are env-driven so docker-compose can override."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8002)
    log_level: str = os.getenv("LOG_LEVEL", "info")
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 500)

    # ── Postgres ──────────────────────────────────────────────────────
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://datahub:datahub@postgres:5432/datahub",
    )

    # ── RabbitMQ / Celery ─────────────────────────────────────────────
    rabbitmq_url: str = os.getenv(
        "RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/"
    )
    celery_queue: str = os.getenv("CELERY_QUEUE", "parse_queue")
    celery_routing_key: str = os.getenv("CELERY_ROUTING_KEY", "parse.document")
    celery_concurrency: int = _env_int("CELERY_CONCURRENCY", 1)

    # ── S3 / MinIO ────────────────────────────────────────────────────
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    s3_region: str = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1")
    s3_access_key: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    s3_secret_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    s3_bucket: str = os.getenv("S3_BUCKET", "document-parsing")
    s3_presigned_ttl: int = _env_int("S3_PRESIGNED_TTL_SECONDS", 3600)
    s3_force_path_style: bool = _env_bool("S3_FORCE_PATH_STYLE", True)

    # ── MinerU vn_parser ──────────────────────────────────────────────
    # Source code is vendored into ``src/vendored/vn_parser`` so the service
    # is self-contained — only the ONNX/torch *weights* are mounted from
    # disk (3.8 GB; not worth baking into the image).
    mineru_models_dir: str = os.getenv(
        "MINERU_MODELS_DIR", "/opt/mineru/models_onnx"
    )
    mineru_enable_vlm: bool = _env_bool("MINERU_ENABLE_VLM", False)
    mineru_vlm_model_path: str | None = os.getenv("MINERU_VLM_MODEL_PATH")
    mineru_vlm_dtype: str = os.getenv("MINERU_VLM_DTYPE", "float32")
    mineru_layout_conf: float = float(os.getenv("MINERU_LAYOUT_CONF", "0.5"))
    mineru_dpi: int = _env_int("MINERU_DPI", 200)
    mineru_vietocr_config: str = os.getenv("MINERU_VIETOCR_CONFIG", "vgg_transformer")

    # ── Per-stage device selection ───────────────────────────────────
    # Values: "auto" | "cpu" | "cuda" | "cuda:N"
    # Sensible CPU-server defaults — flip individual stages to "cuda" or
    # "auto" on a GPU host to accelerate that stage.
    device_layout: str = os.getenv("DEVICE_LAYOUT", "auto")
    device_ocr_det: str = os.getenv("DEVICE_OCR_DET", "auto")
    device_ocr_rec: str = os.getenv("DEVICE_OCR_REC", "auto")
    device_orient: str = os.getenv("DEVICE_ORIENT", "cpu")
    device_table_cls: str = os.getenv("DEVICE_TABLE_CLS", "cpu")
    device_table_rec: str = os.getenv("DEVICE_TABLE_REC", "auto")
    device_mfr: str = os.getenv("DEVICE_MFR", "auto")

    # ── Behavior ──────────────────────────────────────────────────────
    pdf_force_plain: bool = _env_bool("PDF_FORCE_PLAIN", False)
    pdf_hybrid_mode: bool = _env_bool("PDF_HYBRID_MODE", True)
    pdf_hybrid_min_chars: int = _env_int("PDF_HYBRID_MIN_CHARS", 20)
    # Per-block threshold: a text-bearing layout block with fewer pdfplumber
    # words than this falls back to VietOCR for that block alone. Lets us
    # handle mixed pages (born-digital body + scanned diagram with caption).
    pdf_block_min_words: int = _env_int("PDF_BLOCK_MIN_WORDS", 3)
    # Heavy table struct (UNet wired + SLANet wireless + per-cell OCR).
    # On CPU was 20-30 s/table; with batched OCR (see ocr_adapter.py) it
    # comes down to a few seconds. Set false to OCR table regions as plain
    # text without preserving cell structure.
    parse_table_struct: bool = _env_bool("PARSE_TABLE_STRUCT", True)
    # When False, skip the SLANet+ "wireless" table-structure model and
    # always use UNet-wired. Saves the ~600 MB wireless ONNX in RAM and
    # halves table-struct latency on table-heavy pages at the cost of
    # worse output on borderless tables.
    parse_table_wireless: bool = _env_bool("PARSE_TABLE_WIRELESS", False)

    # ── CPU-batched pipeline (TEXT-mode) ─────────────────────────────
    # Master switch: turn off to bypass page-classification + batched
    # path entirely and run the original hybrid pipeline.
    enable_cpu_batched_pipeline: bool = _env_bool("ENABLE_CPU_BATCHED_PIPELINE", True)
    # Document-level: fraction of TEXT-labelled pages required to opt
    # into the batched pipeline. Below this, falls back to SCAN/hybrid.
    doc_text_mode_threshold: float = _env_float("DOC_TEXT_MODE_THRESHOLD", 0.8)
    # Per-page classification thresholds (see page_classifier.classify_pages).
    page_text_min_words: int = _env_int("PAGE_TEXT_MIN_WORDS", 50)
    page_text_coverage_min: float = _env_float("PAGE_TEXT_COVERAGE_MIN", 0.15)
    page_image_max_words: int = _env_int("PAGE_IMAGE_MAX_WORDS", 10)
    page_image_area_ratio: float = _env_float("PAGE_IMAGE_AREA_RATIO", 0.40)
    page_junk_char_threshold: float = _env_float("PAGE_JUNK_CHAR_THRESHOLD", 0.30)
    # When true, SPARSE pages with significant image coverage are
    # reclassified as IMAGE (forces OCR on form-like pages).
    ocr_sparse_with_image: bool = _env_bool("OCR_SPARSE_WITH_IMAGE", False)
    # Batching knobs.
    page_chunk_size: int = _env_int("PAGE_CHUNK_SIZE", 8)
    layout_batch_size: int = _env_int("LAYOUT_BATCH_SIZE", 8)
    dbnet_batch_size: int = _env_int("DBNET_BATCH_SIZE", 8)
    ocr_batch_size: int = _env_int("OCR_BATCH_SIZE", 16)
    ocr_bucket_width_step: int = _env_int("OCR_BUCKET_WIDTH_STEP", 32)
    table_struct_batch_size: int = _env_int("TABLE_STRUCT_BATCH_SIZE", 4)
    # Hint for downstream ONNX session construction; informational only —
    # vendored model wrappers don't currently override their session options.
    onnx_intra_op_threads: int = _env_int("ONNX_INTRA_OP_THREADS", 2)


settings = Settings()
