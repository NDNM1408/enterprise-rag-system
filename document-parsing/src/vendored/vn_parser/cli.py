"""CLI entry: python -m vn_parser <input> -o <output_dir>"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vn_parser.pipeline import VNDocParser


def main():
    p = argparse.ArgumentParser(prog="vn_parser")
    p.add_argument("input", help="Path to PDF or image file")
    p.add_argument("-o", "--output", default="output", help="Output directory")
    p.add_argument("--models", default="models_onnx", help="Directory of ONNX models")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--start", type=int, default=0,
                   help="Start page (0-based, inclusive). Default 0.")
    p.add_argument("--end", type=int, default=None,
                   help="End page (0-based, inclusive). Default = last page.")
    p.add_argument("--no-orient", action="store_true",
                   help="Disable orientation correction")
    p.add_argument("--layout-conf", type=float, default=0.5)
    p.add_argument("--vietocr-config", default="vgg_transformer",
                   help="VietOCR config name (vgg_transformer|vgg_seq2seq|...)")
    p.add_argument("--vietocr-weights", default=None,
                   help="Override VietOCR pretrained weights path")
    p.add_argument("--save-pages", action="store_true",
                   help="Also save layout-visualized page PNGs")
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading models from {args.models}...")
    parser = VNDocParser(
        models_dir=args.models,
        enable_orientation=not args.no_orient,
        layout_conf=args.layout_conf,
        vietocr_config=args.vietocr_config,
        vietocr_weights=args.vietocr_weights,
    )

    print(f"Parsing {in_path} ...")
    pages = parser.load_pages(in_path, dpi=args.dpi)
    end = args.end if args.end is not None else len(pages) - 1
    pages = pages[args.start:end + 1]
    print(f"  rendering pages {args.start}..{end} ({len(pages)} pages)")
    results = []
    import time
    for i, p in enumerate(pages):
        page_idx = args.start + i
        t0 = time.time()
        results.append(parser.parse_page(p, page_index=page_idx))
        print(f"  page {page_idx}: {len(results[-1].blocks)} blocks "
              f"({time.time() - t0:.1f}s)")

    md_path = out_dir / (in_path.stem + ".md")
    json_path = out_dir / (in_path.stem + ".json")
    md_path.write_text(parser.to_markdown(results), encoding="utf-8")
    json_path.write_text(
        json.dumps(parser.to_json(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {md_path}")
    print(f"  wrote {json_path}")

    if args.save_pages:
        from vn_parser.visualization import draw_layout
        for i, (img, res) in enumerate(zip(pages, results)):
            blocks = [
                {"label": b.label, "bbox": list(b.bbox),
                 "index": b.index, "score": b.score}
                for b in res.blocks
            ]
            overlay = draw_layout(img, blocks)
            page_no = args.start + i + 1
            out_path = out_dir / f"{in_path.stem}_page{page_no}.png"
            overlay.save(out_path)
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
