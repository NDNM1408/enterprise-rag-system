"""Wired table recognition (UNet) — vendored from mineru-source.

Adapter API expected for `ocr_engine`:
    ocr(img, det=True)  -> [[(bbox_4x2, (text, score)), ...]]
    ocr(img_list, det=False) -> [[(text, score), ...]]
"""
from __future__ import annotations

import html
import logging
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from loguru import logger

from .table_recover import TableRecover
from .table_structure_unet import TSRUnet
from .utils import InputType, LoadImage
from .utils_table_recover import (
    box_4_2_poly_to_box_4_1,
    gather_ocr_list_by_row,
    match_ocr_cell,
    plot_html_table,
    sorted_ocr_boxes,
)


def _calculate_contrast(img: np.ndarray, img_mode: str = "bgr") -> float:
    """Pure-Python port of mineru.utils.span_pre_proc.calculate_contrast."""
    if img is None or img.size == 0:
        return 0.0
    if img.ndim == 2:
        gray = img
    else:
        cvt = cv2.COLOR_RGB2GRAY if img_mode == "rgb" else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(img, cvt)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    return std / (mean + 1e-6)


@dataclass
class WiredTableInput:
    model_path: str
    device: str = "cpu"


@dataclass
class WiredTableOutput:
    pred_html: Optional[str] = None
    cell_bboxes: Optional[np.ndarray] = None
    logic_points: Optional[np.ndarray] = None
    elapse: Optional[float] = None


class WiredTableRecognition:
    """Run wired-table structure model + OCR matching to produce HTML.

    Mirrors MinerU's WiredTableRecognition (mineru-source/.../unet_table/main.py)
    but with mineru.utils dependencies stubbed locally.
    """

    def __init__(self, config: WiredTableInput, ocr_engine=None):
        self.table_structure = TSRUnet(asdict(config))
        self.load_img = LoadImage()
        self.table_recover = TableRecover()
        self.ocr_engine = ocr_engine

    def __call__(
        self,
        img: InputType,
        ocr_result: Optional[List[Any]] = None,
        **kwargs,
    ) -> WiredTableOutput:
        s = time.perf_counter()
        need_ocr = kwargs.get("need_ocr", True)
        col_threshold = kwargs.get("col_threshold", 15)
        row_threshold = kwargs.get("row_threshold", 10)

        img = self.load_img(img)
        polygons, rotated_polygons = self.table_structure(img, **kwargs)
        if polygons is None:
            return WiredTableOutput("", None, None, 0.0)

        try:
            table_res, logi_points = self.table_recover(
                rotated_polygons, row_threshold, col_threshold
            )
            polygons[:, 1, :], polygons[:, 3, :] = (
                polygons[:, 3, :].copy(),
                polygons[:, 1, :].copy(),
            )
            if not need_ocr:
                sorted_polygons, idx_list = sorted_ocr_boxes(
                    [box_4_2_poly_to_box_4_1(box) for box in polygons]
                )
                return WiredTableOutput(
                    "", sorted_polygons, logi_points[idx_list],
                    time.perf_counter() - s,
                )

            cell_box_det_map, _ = match_ocr_cell(ocr_result or [], polygons)
            cell_box_det_map = self._fill_blank_rec(img, polygons, cell_box_det_map)
            t_rec_ocr_list = self._transform_res(cell_box_det_map, polygons, logi_points)
            t_rec_ocr_list = self._sort_and_gather_ocr_res(t_rec_ocr_list)

            logi_points = [t["t_logic_box"] for t in t_rec_ocr_list]
            cell_box_det_map = {
                i: [pair[1] for pair in t["t_ocr_res"]]
                for i, t in enumerate(t_rec_ocr_list)
            }
            pred_html = plot_html_table(logi_points, cell_box_det_map)
            polygons = np.array(polygons).reshape(-1, 8)
            logi_points = np.array(logi_points)
            elapse = time.perf_counter() - s
        except Exception:
            logging.warning(traceback.format_exc())
            return WiredTableOutput("", None, None, 0.0)
        return WiredTableOutput(pred_html, polygons, logi_points, elapse)

    @staticmethod
    def _transform_res(
        cell_box_det_map: Dict[int, List[Any]],
        polygons: np.ndarray,
        logi_points: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        res = []
        for i in range(len(polygons)):
            ocr_res_list = cell_box_det_map.get(i)
            if not ocr_res_list:
                continue
            xmin = min([o[0][0][0] for o in ocr_res_list])
            ymin = min([o[0][0][1] for o in ocr_res_list])
            xmax = max([o[0][2][0] for o in ocr_res_list])
            ymax = max([o[0][2][1] for o in ocr_res_list])
            res.append({
                "t_box": [xmin, ymin, xmax, ymax],
                "t_logic_box": logi_points[i].tolist(),
                "t_ocr_res": [
                    [box_4_2_poly_to_box_4_1(o[0]), o[1]] for o in ocr_res_list
                ],
            })
        return res

    @staticmethod
    def _sort_and_gather_ocr_res(res):
        for dict_res in res:
            _, sorted_idx = sorted_ocr_boxes(
                [o[0] for o in dict_res["t_ocr_res"]], threhold=0.3
            )
            dict_res["t_ocr_res"] = [dict_res["t_ocr_res"][i] for i in sorted_idx]
            dict_res["t_ocr_res"] = gather_ocr_list_by_row(
                dict_res["t_ocr_res"], threhold=0.3
            )
        return res

    def _fill_blank_rec(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[Any]],
    ) -> Dict[int, List[Any]]:
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_crop_info_list: List[List[Any]] = []
        img_crop_list: List[np.ndarray] = []
        for i in range(sorted_polygons.shape[0]):
            if cell_box_map.get(i):
                continue
            box = sorted_polygons[i]
            if self.ocr_engine is None:
                continue
            x1 = int(box[0][0]) + 1
            y1 = int(box[0][1]) + 1
            x2 = int(box[2][0]) - 1
            y2 = int(box[2][1]) - 1
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                continue
            if (x2 - x1) / max(y2 - y1, 1) > 20 or (y2 - y1) / max(x2 - x1, 1) > 20:
                continue
            img_crop = bgr_img[y1:y2, x1:x2]
            if _calculate_contrast(img_crop, img_mode="bgr") <= 0.17:
                cell_box_map[i] = [[box, "", 0.1]]
                continue
            img_crop_list.append(img_crop)
            img_crop_info_list.append([i, box])

        if not img_crop_list:
            return cell_box_map

        ocr_result = self.ocr_engine.ocr(img_crop_list, det=False)
        if not ocr_result or not isinstance(ocr_result, list):
            return cell_box_map
        ocr_res_list = ocr_result[0]
        if not isinstance(ocr_res_list, list) or len(ocr_res_list) != len(img_crop_list):
            return cell_box_map
        for j, ocr_res in enumerate(ocr_res_list):
            img_crop_info_list[j].append(ocr_res)

        for i, box, ocr_res in img_crop_info_list:
            ocr_text, ocr_score = ocr_res
            if ocr_score < 0.6 or ocr_text in {
                "1", "口", "■",
            }:
                cell_box_map[i] = [[box, "", 0.1]]
                continue
            cell_box_map[i] = [[box, ocr_text, ocr_score]]
        return cell_box_map


def escape_html(s: str) -> str:
    return html.escape(s)


class UnetWiredTable:
    """Convenience top-level class — runs wired-only (no wireless switching).

    Use this directly from pipeline.py.
    """
    def __init__(self, model_path: str, ocr_engine):
        self.wired = WiredTableRecognition(WiredTableInput(model_path=model_path), ocr_engine)
        self.ocr_engine = ocr_engine

    def predict(self, img_rgb: Union[np.ndarray, Image.Image]) -> str:
        if isinstance(img_rgb, Image.Image):
            img_rgb = np.asarray(img_rgb)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ocr_result = self.ocr_engine.ocr(bgr)[0]
        ocr_result = [
            [item[0], escape_html(item[1][0]), item[1][1]]
            for item in ocr_result
            if len(item) == 2 and isinstance(item[1], tuple)
        ]
        result = self.wired(img_rgb, ocr_result)
        return result.pred_html or ""
