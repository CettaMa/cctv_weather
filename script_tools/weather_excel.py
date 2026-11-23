#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Excel exporter for CCTV weather classification benchmark.
Consolidates all benchmark sections into a single, vertically structured sheet ("Benchmark"):
  1. Title & Executive Callout
  2. Model Leaderboard (Accuracy, Macro metrics, ROC AUC, Latency)
  3. Visual Comparison Charts (Accuracy vs Macro F1, Per-class F1, ROC AUC, Latency)
  4. Per-Class Metrics (Precision, Recall, F1, Support)
  5. Confusion Matrices (2x2 matrices with white->blue heatmaps)
  6. Feature Importance (Normalized tabular feature importances + horizontal bar chart)
  7. Neural Training Progression (Epoch loss & validation accuracy curves)
  8. Model Hyperparameters & Configurations
  9. Methodology, Dataset & Caveats Notes
"""
from __future__ import annotations

import datetime
import math
from pathlib import Path

from script_tools.weather_tabular import TAB_FEATURES, TAB_FEATURE_DESC


def _abort(msg: str) -> None:
    import sys
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)

def write_workbook(records: list[dict], path: Path | str, ds=None) -> None:
    """Write comprehensive benchmark results to a single vertical Excel sheet."""
    if ds is None:
        from script_tools.weather_ml import DATASETS
        ds = DATASETS["sp_buahbatu"]
    try:
        import xlsxwriter
    except ImportError:
        _abort("xlsxwriter not installed (.venv/Scripts/python.exe -m pip install xlsxwriter)")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(str(out_path), {"nan_inf_to_errors": True})

    # -------------------------------------------------------------------------
    # Visual Palette & Formats
    # -------------------------------------------------------------------------
    BLUE = "#2563EB"
    GRAY = "#64748B"
    MACRO = "#0EA5E9"
    HEADER_BG = "#1E3A8A"
    SECTION_BG = "#0F172A"
    BEST_BG = "#DCFCE7"

    fmt_title = workbook.add_format({
        "font_name": "Calibri", "font_size": 16, "bold": True, "font_color": HEADER_BG
    })
    fmt_subtitle = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "italic": True, "font_color": "#475569"
    })
    fmt_best_callout = workbook.add_format({
        "font_name": "Calibri", "font_size": 11, "bold": True, "font_color": "#166534",
        "bg_color": BEST_BG, "border": 1, "border_color": "#86EFAC"
    })
    fmt_section_hdr = workbook.add_format({
        "font_name": "Calibri", "font_size": 12, "bold": True, "font_color": "white",
        "bg_color": SECTION_BG, "align": "left", "valign": "vcenter", "indent": 1
    })
    fmt_th = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "bold": True, "font_color": "white",
        "bg_color": HEADER_BG, "align": "center", "valign": "vcenter", "text_wrap": True, "border": 1
    })
    fmt_th_sub = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "bold": True, "font_color": HEADER_BG,
        "bg_color": "#E2E8F0", "align": "center", "valign": "vcenter", "border": 1
    })
    fmt_cell = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "border": 1, "border_color": "#CBD5E1"
    })
    fmt_cell_center = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "align": "center", "border": 1, "border_color": "#CBD5E1"
    })
    fmt_cell_bold = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "bold": True, "border": 1, "border_color": "#CBD5E1"
    })
    fmt_metric = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "num_format": "0.000", "align": "right",
        "border": 1, "border_color": "#CBD5E1"
    })
    fmt_seconds = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "num_format": "0.00", "align": "right",
        "border": 1, "border_color": "#CBD5E1"
    })
    fmt_ms = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "num_format": "0.000", "align": "right",
        "border": 1, "border_color": "#CBD5E1"
    })
    fmt_int = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "num_format": "#,##0", "align": "right",
        "border": 1, "border_color": "#CBD5E1"
    })
    fmt_cm_header = workbook.add_format({
        "font_name": "Calibri", "font_size": 10, "bold": True, "align": "center",
        "bg_color": "#F1F5F9", "border": 1
    })
    fmt_cm_num = workbook.add_format({
        "font_name": "Calibri", "font_size": 11, "bold": True, "align": "center",
        "valign": "vcenter", "border": 1
    })
    fmt_params = workbook.add_format({
        "font_name": "Consolas", "font_size": 9, "text_wrap": True, "valign": "top",
        "border": 1, "border_color": "#CBD5E1"
    })

    # Sort records for Leaderboard
    sorted_records = sorted(
        records,
        key=lambda r: (
            r.get("macro", {}).get("f1", 0.0),
            r.get("accuracy", 0.0),
            r.get("roc_auc", 0.0),
            -r.get("train_seconds", 0.0),
        ),
        reverse=True,
    )

    n_train = sorted_records[0].get("n_train", 192) if sorted_records else 192
    n_test = sorted_records[0].get("n_test", 48) if sorted_records else 48
    best = sorted_records[0] if sorted_records else {}

    ws = workbook.add_worksheet("Benchmark")
    ws.hide_gridlines(2)

    # Global column widths
    ws.set_column(0, 0, 7)    # Col A: Rank / Index / Bullets
    ws.set_column(1, 1, 18)   # Col B: Model / Feature name / Conf Mat
    ws.set_column(2, 2, 18)   # Col C: Family / Metric / Conf Mat
    ws.set_column(3, 3, 22)   # Col D: Input / Conf Mat
    ws.set_column(4, 4, 13)   # Col E: Accuracy
    ws.set_column(5, 5, 13)   # Col F: Precision (macro)
    ws.set_column(6, 6, 13)   # Col G: Recall (macro)
    ws.set_column(7, 7, 13)   # Col H: F1 (macro)
    ws.set_column(8, 8, 13)   # Col I: F1 Rain
    ws.set_column(9, 9, 13)   # Col J: F1 Sunny
    ws.set_column(10, 10, 13) # Col K: ROC AUC
    ws.set_column(11, 11, 14) # Col L: Train time (s)
    ws.set_column(12, 12, 18) # Col M: Infer (ms/sample)
    ws.set_column(13, 14, 16) # Col N-O: Extra feature importance cols

    # =========================================================================
    # 1. Header & Executive Summary
    # =========================================================================
    ws.write(0, 0, "CCTV Weather Classification — Rain vs Sunny Benchmark", fmt_title)
    ws.write(
        1, 0,
        f"Camera: {ds.camera} | Train: {n_train} frames | Test: {n_test} frames (balanced) | Positive class: Rain",
        fmt_subtitle,
    )
    ws.write(2, 0, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", fmt_subtitle)

    best_name = best.get("display_name", "N/A")
    best_mf1 = best.get("macro", {}).get("f1", 0.0)
    best_acc = best.get("accuracy", 0.0)
    best_auc = best.get("roc_auc", 0.0)
    callout_txt = f"  Best Model: {best_name} — Macro F1: {best_mf1:.3f}, Accuracy: {best_acc:.3f}, ROC AUC: {best_auc:.3f}  "
    ws.write(4, 0, callout_txt, fmt_best_callout)

    # =========================================================================
    # 2. Section: Model Leaderboard
    # =========================================================================
    ws.merge_range(6, 0, 6, 12, "1. Overall Performance Leaderboard", fmt_section_hdr)

    sum_cols = [
        "Rank", "Model", "Family", "Input Modality", "Accuracy", "Precision (macro)",
        "Recall (macro)", "F1 (macro)", "F1 Rain", "F1 Sunny", "ROC AUC",
        "Train time (s)", "Inference (ms/sample)",
    ]

    lb_data = []
    for rank, r in enumerate(sorted_records, 1):
        row = [
            rank,
            r.get("display_name", r.get("model", "")),
            r.get("family", ""),
            r.get("input", ""),
            r.get("accuracy", 0.0),
            r.get("macro", {}).get("precision", 0.0),
            r.get("macro", {}).get("recall", 0.0),
            r.get("macro", {}).get("f1", 0.0),
            r.get("per_class", {}).get("rain", {}).get("f1", 0.0),
            r.get("per_class", {}).get("sunny", {}).get("f1", 0.0),
            r.get("roc_auc", 0.0),
            r.get("train_seconds", 0.0),
            r.get("infer_ms_per_sample", 0.0),
        ]
        lb_data.append(row)

    lb_start_row = 8
    lb_end_row = lb_start_row + len(lb_data)

    ws.add_table(
        lb_start_row, 0, lb_end_row, len(sum_cols) - 1,
        {
            "data": lb_data,
            "columns": [{"header": c} for c in sum_cols],
            "style": "Table Style Medium 2",
        },
    )

    for r_idx, row in enumerate(lb_data):
        curr_row = lb_start_row + 1 + r_idx
        ws.write(curr_row, 0, row[0], fmt_cell_center)
        ws.write(curr_row, 1, row[1], fmt_cell_bold)
        ws.write(curr_row, 2, row[2], fmt_cell)
        ws.write(curr_row, 3, row[3], fmt_cell)
        for c_idx in range(4, 11):
            ws.write(curr_row, c_idx, row[c_idx], fmt_metric)
        ws.write(curr_row, 11, row[11], fmt_seconds)
        ws.write(curr_row, 12, row[12], fmt_ms)

    for c_idx in range(4, 11):
        col_letter = chr(ord('A') + c_idx)
        rng = f"{col_letter}{lb_start_row + 2}:{col_letter}{lb_end_row + 1}"
        ws.conditional_format(rng, {
            "type": "data_bar",
            "bar_color": BLUE,
            "min_type": "num", "min_value": 0.0,
            "max_type": "num", "max_value": 1.0,
        })
        ws.conditional_format(rng, {
            "type": "top",
            "value": 1,
            "format": workbook.add_format({"bold": True, "bg_color": BEST_BG}),
        })

    # =========================================================================
    # 3. Section: Benchmark Charts
    # =========================================================================
    charts_hdr_row = lb_end_row + 3
    ws.merge_range(charts_hdr_row, 0, charts_hdr_row, 12, "2. Benchmark Comparison Charts", fmt_section_hdr)

    n_models = len(sorted_records)
    r_first = lb_start_row + 2
    r_last = lb_start_row + 1 + n_models

    min_acc_f1 = min(
        [r.get("accuracy", 0.0) for r in sorted_records] +
        [r.get("macro", {}).get("f1", 0.0) for r in sorted_records] or [0.5]
    )
    y_min_val = max(0.0, math.floor((min_acc_f1 - 0.05) * 10) / 10.0)

    # Chart 1: Accuracy vs Macro F1
    chart1 = workbook.add_chart({"type": "column"})
    chart1.add_series({
        "name": f"=Benchmark!$E${lb_start_row + 1}",
        "categories": f"=Benchmark!$B${r_first}:$B${r_last}",
        "values": f"=Benchmark!$E${r_first}:$E${r_last}",
        "fill": {"color": BLUE},
        "data_labels": {"value": True, "num_format": "0.000", "font": {"size": 8}},
    })
    chart1.add_series({
        "name": f"=Benchmark!$H${lb_start_row + 1}",
        "categories": f"=Benchmark!$B${r_first}:$B${r_last}",
        "values": f"=Benchmark!$H${r_first}:$H${r_last}",
        "fill": {"color": MACRO},
        "data_labels": {"value": True, "num_format": "0.000", "font": {"size": 8}},
    })
    chart1.set_title({"name": "Accuracy vs Macro F1 Score"})
    chart1.set_y_axis({"min": y_min_val, "max": 1.0, "major_gridlines": {"visible": True}})
    chart1.set_size({"width": 560, "height": 330})
    ws.insert_chart(f"B{charts_hdr_row + 2}", chart1)

    # Chart 2: Per-class F1
    chart2 = workbook.add_chart({"type": "column"})
    chart2.add_series({
        "name": f"=Benchmark!$I${lb_start_row + 1}",
        "categories": f"=Benchmark!$B${r_first}:$B${r_last}",
        "values": f"=Benchmark!$I${r_first}:$I${r_last}",
        "fill": {"color": BLUE},
        "data_labels": {"value": True, "num_format": "0.000", "font": {"size": 8}},
    })
    chart2.add_series({
        "name": f"=Benchmark!$J${lb_start_row + 1}",
        "categories": f"=Benchmark!$B${r_first}:$B${r_last}",
        "values": f"=Benchmark!$J${r_first}:$J${r_last}",
        "fill": {"color": GRAY},
        "data_labels": {"value": True, "num_format": "0.000", "font": {"size": 8}},
    })
    chart2.set_title({"name": "Per-Class F1 (Rain vs Sunny)"})
    chart2.set_y_axis({"min": y_min_val, "max": 1.0, "major_gridlines": {"visible": True}})
    chart2.set_size({"width": 560, "height": 330})
    ws.insert_chart(f"H{charts_hdr_row + 2}", chart2)

    # Chart 3: ROC AUC
    chart3 = workbook.add_chart({"type": "bar"})
    chart3.add_series({
        "name": f"=Benchmark!$K${lb_start_row + 1}",
        "categories": f"=Benchmark!$B${r_first}:$B${r_last}",
        "values": f"=Benchmark!$K${r_first}:$K${r_last}",
        "fill": {"color": BLUE},
        "data_labels": {"value": True, "num_format": "0.000", "font": {"size": 8}},
    })
    chart3.set_title({"name": "ROC AUC"})
    chart3.set_x_axis({"min": y_min_val, "max": 1.0, "major_gridlines": {"visible": True}})
    chart3.set_y_axis({"reverse_categories": True})
    chart3.set_legend({"none": True})
    chart3.set_size({"width": 560, "height": 310})
    ws.insert_chart(f"B{charts_hdr_row + 20}", chart3)

    # Chart 4: Training time (s)
    chart4 = workbook.add_chart({"type": "bar"})
    chart4.add_series({
        "name": f"=Benchmark!$L${lb_start_row + 1}",
        "categories": f"=Benchmark!$B${r_first}:$B${r_last}",
        "values": f"=Benchmark!$L${r_first}:$L${r_last}",
        "fill": {"color": GRAY},
        "data_labels": {"value": True, "num_format": "0.00", "font": {"size": 8}},
    })
    chart4.set_title({"name": "Training Time (seconds)"})
    chart4.set_x_axis({"log_base": 10, "major_gridlines": {"visible": True}})
    chart4.set_y_axis({"reverse_categories": True})
    chart4.set_legend({"none": True})
    chart4.set_size({"width": 560, "height": 310})
    ws.insert_chart(f"H{charts_hdr_row + 20}", chart4)

    # =========================================================================
    # 4. Section: Per-Class Metrics Table
    # =========================================================================
    pc_hdr_row = charts_hdr_row + 38
    ws.merge_range(pc_hdr_row, 0, pc_hdr_row, 12, "3. Per-Class Precision, Recall, F1 & Support", fmt_section_hdr)

    pc_headers = ["Model", "Family", "Class", "Precision", "Recall", "F1 Score", "Support"]
    pc_data = []
    for r in sorted_records:
        m_name = r.get("display_name", r.get("model", ""))
        fam = r.get("family", "")
        cm = r.get("confusion", [[0, 0], [0, 0]])
        sup_rain = sum(cm[0]) if len(cm) > 0 else 0
        sup_sunny = sum(cm[1]) if len(cm) > 1 else 0

        p_rain = r.get("per_class", {}).get("rain", {}).get("precision", 0.0)
        rc_rain = r.get("per_class", {}).get("rain", {}).get("recall", 0.0)
        f1_rain = r.get("per_class", {}).get("rain", {}).get("f1", 0.0)

        p_sunny = r.get("per_class", {}).get("sunny", {}).get("precision", 0.0)
        rc_sunny = r.get("per_class", {}).get("sunny", {}).get("recall", 0.0)
        f1_sunny = r.get("per_class", {}).get("sunny", {}).get("f1", 0.0)

        pc_data.append([m_name, fam, "Rain", p_rain, rc_rain, f1_rain, sup_rain])
        pc_data.append([m_name, fam, "Sunny", p_sunny, rc_sunny, f1_sunny, sup_sunny])

    pc_start = pc_hdr_row + 2
    pc_end = pc_start + len(pc_data)
    ws.add_table(
        pc_start, 1, pc_end, 1 + len(pc_headers) - 1,
        {
            "data": pc_data,
            "columns": [{"header": h} for h in pc_headers],
            "style": "Table Style Medium 2",
        },
    )

    for idx, row in enumerate(pc_data):
        curr_row = pc_start + 1 + idx
        ws.write(curr_row, 1, row[0], fmt_cell_bold)
        ws.write(curr_row, 2, row[1], fmt_cell)
        ws.write(curr_row, 3, row[2], fmt_cell_center)
        ws.write(curr_row, 4, row[3], fmt_metric)
        ws.write(curr_row, 5, row[4], fmt_metric)
        ws.write(curr_row, 6, row[5], fmt_metric)
        ws.write(curr_row, 7, row[6], fmt_int)

    for c_idx in range(4, 7):
        col_letter = chr(ord('A') + c_idx)
        rng = f"{col_letter}{pc_start + 2}:{col_letter}{pc_end + 1}"
        ws.conditional_format(rng, {
            "type": "data_bar",
            "bar_color": BLUE,
            "min_type": "num", "min_value": 0.0,
            "max_type": "num", "max_value": 1.0,
        })

    # =========================================================================
    # 5. Section: Confusion Matrices
    # =========================================================================
    cm_hdr_row = pc_end + 3
    ws.merge_range(cm_hdr_row, 0, cm_hdr_row, 12, f"4. Confusion Matrices (Test Set: {n_test} Balanced Frames)", fmt_section_hdr)
    cm_col_offsets = [1, 6]  # Cols B and G
    for idx, r in enumerate(sorted_records):
        block_row = cm_hdr_row + 2 + (idx // 2) * 8
        b_col = cm_col_offsets[idx % 2]
        m_name = r.get("display_name", r.get("model", ""))
        rank = idx + 1
        cm = r.get("confusion", [[0, 0], [0, 0]])
        acc = r.get("accuracy", 0.0)

        # Title
        ws.merge_range(block_row, b_col, block_row, b_col + 2, f"{rank}. {m_name}", fmt_th)
        # Header
        ws.write(block_row + 1, b_col, "Actual \\ Pred", fmt_cm_header)
        ws.write(block_row + 1, b_col + 1, "Rain (0)", fmt_cm_header)
        ws.write(block_row + 1, b_col + 2, "Sunny (1)", fmt_cm_header)
        # Row 1: Rain
        ws.write(block_row + 2, b_col, "Rain", fmt_cm_header)
        ws.write(block_row + 2, b_col + 1, cm[0][0] if len(cm) > 0 else 0, fmt_cm_num)
        ws.write(block_row + 2, b_col + 2, cm[0][1] if len(cm) > 0 else 0, fmt_cm_num)
        # Row 2: Sunny
        ws.write(block_row + 3, b_col, "Sunny", fmt_cm_header)
        ws.write(block_row + 3, b_col + 1, cm[1][0] if len(cm) > 1 else 0, fmt_cm_num)
        ws.write(block_row + 3, b_col + 2, cm[1][1] if len(cm) > 1 else 0, fmt_cm_num)
        # Footer
        ws.merge_range(block_row + 4, b_col, block_row + 4, b_col + 2, f"Accuracy: {acc:.3f}", fmt_th_sub)

        # 2-color heatmap on 2x2 cells
        c1 = chr(ord('A') + b_col + 1)
        c2 = chr(ord('A') + b_col + 2)
        r1 = block_row + 3
        r2 = block_row + 4
        ws.conditional_format(f"{c1}{r1}:{c2}{r2}", {
            "type": "2_color_scale",
            "min_color": "#FFFFFF",
            "max_color": "#BFDBFE",
            "min_type": "num", "min_value": 0,
            "max_type": "num", "max_value": 24,
        })

    total_cm_rows = ((len(sorted_records) + 1) // 2) * 8

    # =========================================================================
    # 6. Section: Feature Importance
    # =========================================================================
    fi_hdr_row = cm_hdr_row + 2 + total_cm_rows + 2
    ws.merge_range(fi_hdr_row, 0, fi_hdr_row, 12, "5. Tabular Feature Importance Comparison", fmt_section_hdr)

    fi_models = [r for r in records if r.get("feature_importance")]
    fi_headers = ["Feature", "Description"] + [r["display_name"] for r in fi_models] + ["Mean Importance"]

    fi_rows = []
    for feat in TAB_FEATURES:
        desc = TAB_FEATURE_DESC.get(feat, "")
        vals = [float(r["feature_importance"].get(feat, 0.0)) for r in fi_models]
        mean_v = float(sum(vals) / len(vals)) if vals else 0.0
        fi_rows.append([feat, desc] + vals + [mean_v])

    fi_rows.sort(key=lambda x: x[-1], reverse=True)

    fi_start = fi_hdr_row + 2
    fi_end = fi_start + len(fi_rows)
    ws.add_table(
        fi_start, 1, fi_end, 1 + len(fi_headers) - 1,
        {
            "data": fi_rows,
            "columns": [{"header": h} for h in fi_headers],
            "style": "Table Style Medium 2",
        },
    )

    for idx, row in enumerate(fi_rows):
        curr_row = fi_start + 1 + idx
        ws.write(curr_row, 1, row[0], fmt_cell_bold)
        ws.write(curr_row, 2, row[1], fmt_cell)
        for c_idx in range(2, len(fi_headers)):
            ws.write(curr_row, 1 + c_idx, row[c_idx], fmt_metric)

    for c_idx in range(2, len(fi_headers)):
        col_letter = chr(ord('A') + 1 + c_idx)
        rng = f"{col_letter}{fi_start + 2}:{col_letter}{fi_end + 1}"
        ws.conditional_format(rng, {
            "type": "data_bar",
            "bar_color": BLUE,
        })

    # Horizontal bar chart of Mean Feature Importance
    chart_fi = workbook.add_chart({"type": "bar"})
    mean_col_letter = chr(ord('A') + 1 + len(fi_headers) - 1)
    chart_fi.add_series({
        "name": f"=Benchmark!${mean_col_letter}${fi_start + 1}",
        "categories": f"=Benchmark!$B${fi_start + 2}:$B${fi_end + 1}",
        "values": f"=Benchmark!${mean_col_letter}${fi_start + 2}:${mean_col_letter}${fi_end + 1}",
        "fill": {"color": BLUE},
        "data_labels": {"value": True, "num_format": "0.000", "font": {"size": 8}},
    })
    chart_fi.set_title({"name": "Mean Feature Importance (Tree Models + TabNet)"})
    chart_fi.set_y_axis({"reverse_categories": True})
    chart_fi.set_legend({"none": True})
    chart_fi.set_size({"width": 820, "height": 520})
    ws.insert_chart(f"B{fi_end + 3}", chart_fi)

    # =========================================================================
    # 7. Section: Neural Training Progression (Training Curves)
    # =========================================================================
    tc_hdr_row = fi_end + 33
    ws.merge_range(tc_hdr_row, 0, tc_hdr_row, 12, "6. Neural Network Training Progression (Loss & Validation Accuracy)", fmt_section_hdr)

    dl_records = [r for r in records if r.get("history")]
    tc_start_row = tc_hdr_row + 2
    col_block = 0
    max_epochs_found = 0

    for r in dl_records:
        m_name = r.get("display_name", r.get("model", ""))
        hist = r["history"]
        n_epochs = len(hist.get("train_loss", []))
        if n_epochs == 0:
            continue
        max_epochs_found = max(max_epochs_found, n_epochs)

        c_offset = 1 + col_block * 5
        ws.merge_range(tc_start_row, c_offset, tc_start_row, c_offset + 3, m_name, fmt_th)
        headers = ["Epoch", "Train Loss", "Val Loss", "Val Acc"]
        for j, h in enumerate(headers):
            ws.write(tc_start_row + 1, c_offset + j, h, fmt_th_sub)

        for ep in range(n_epochs):
            curr_r = tc_start_row + 2 + ep
            ws.write(curr_r, c_offset, ep + 1, fmt_cell_center)
            ws.write(curr_r, c_offset + 1, hist["train_loss"][ep], fmt_metric)
            ws.write(curr_r, c_offset + 2, hist["val_loss"][ep], fmt_metric)
            ws.write(curr_r, c_offset + 3, hist["val_acc"][ep], fmt_metric)

        # Line chart for this neural model
        first_r = tc_start_row + 3
        last_r = tc_start_row + 2 + n_epochs
        c_train = chr(ord('A') + c_offset + 1)
        c_val = chr(ord('A') + c_offset + 2)
        c_acc = chr(ord('A') + c_offset + 3)
        c_ep = chr(ord('A') + c_offset)

        chart_l = workbook.add_chart({"type": "line"})
        chart_l.add_series({
            "name": f"=Benchmark!${c_train}${tc_start_row + 2}",
            "categories": f"=Benchmark!${c_ep}${first_r}:${c_ep}${last_r}",
            "values": f"=Benchmark!${c_train}${first_r}:${c_train}${last_r}",
            "line": {"color": BLUE, "width": 1.75},
        })
        chart_l.add_series({
            "name": f"=Benchmark!${c_val}${tc_start_row + 2}",
            "categories": f"=Benchmark!${c_ep}${first_r}:${c_ep}${last_r}",
            "values": f"=Benchmark!${c_val}${first_r}:${c_val}${last_r}",
            "line": {"color": GRAY, "width": 1.75},
        })
        chart_l.add_series({
            "name": f"=Benchmark!${c_acc}${tc_start_row + 2}",
            "categories": f"=Benchmark!${c_ep}${first_r}:${c_ep}${last_r}",
            "values": f"=Benchmark!${c_acc}${first_r}:${c_acc}${last_r}",
            "line": {"color": MACRO, "width": 1.75},
            "y2_axis": True,
        })
        chart_l.set_title({"name": f"{m_name} — Loss & Validation Accuracy"})
        chart_l.set_x_axis({"name": "Epoch"})
        chart_l.set_y_axis({"name": "Cross-Entropy Loss"})
        chart_l.set_y2_axis({"name": "Val Accuracy", "min": 0.0, "max": 1.0})
        chart_l.set_size({"width": 540, "height": 300})

        # Insert charts side by side below the tables
        chart_row_pos = tc_start_row + max(max_epochs_found + 4, 30)
        c_letter = "B" if (col_block % 2 == 0) else "H"
        r_offset_chart = chart_row_pos + (col_block // 2) * 17
        ws.insert_chart(f"{c_letter}{r_offset_chart}", chart_l)

        col_block += 1

    total_tc_rows = max(max_epochs_found + 4, 30) + ((len(dl_records) + 1) // 2) * 17

    # =========================================================================
    # 8. Section: Hyperparameters
    # =========================================================================
    hp_hdr_row = tc_start_row + total_tc_rows + 3
    ws.merge_range(hp_hdr_row, 0, hp_hdr_row, 12, "7. Model Configurations & Hyperparameters", fmt_section_hdr)

    hp_headers = ["Model", "Family", "Library", "Input Modality", "Parameters"]
    hp_data = []
    for r in sorted_records:
        m_name = r.get("display_name", r.get("model", ""))
        fam = r.get("family", "")
        lib = r.get("library", "")
        inp = r.get("input", "")
        params_dict = r.get("params", {})
        param_str = "; ".join(f"{k}={v}" for k, v in params_dict.items())
        hp_data.append([m_name, fam, lib, inp, param_str])

    hp_start = hp_hdr_row + 2
    hp_end = hp_start + len(hp_data)
    ws.add_table(
        hp_start, 1, hp_end, 1 + len(hp_headers) - 1,
        {
            "data": hp_data,
            "columns": [{"header": h} for h in hp_headers],
            "style": "Table Style Medium 2",
        },
    )

    for idx, row in enumerate(hp_data):
        curr_row = hp_start + 1 + idx
        ws.write(curr_row, 1, row[0], fmt_cell_bold)
        ws.write(curr_row, 2, row[1], fmt_cell)
        ws.write(curr_row, 3, row[2], fmt_cell)
        ws.write(curr_row, 4, row[3], fmt_cell)
        ws.write(curr_row, 5, row[4], fmt_params)

    # =========================================================================
    # 9. Section: Notes, Dataset & Reproduction
    # =========================================================================
    notes_hdr_row = hp_end + 3
    ws.merge_range(notes_hdr_row, 0, notes_hdr_row, 12, "8. Methodology, Dataset & Reproduction Notes", fmt_section_hdr)

    sections = [
        ("A. Metric Definitions", [
            "• Accuracy: Overall proportion of correct classifications on the test set: (TP + TN) / (TP + TN + FP + FN).",
            "• Precision: Proportion of predicted positives that were truly positive: TP / (TP + FP).",
            "• Recall: Proportion of actual positives captured by the classifier: TP / (TP + FN).",
            "• F1 Score: Harmonic mean of precision and recall: 2 * (Precision * Recall) / (Precision + Recall).",
            "• Macro F1: Unweighted arithmetic mean of F1 scores across Rain and Sunny classes.",
            "• ROC AUC: Area under the Receiver Operating Characteristic curve computed on predicted rain probabilities.",
            "• Confusion Matrix Layout: Row 1 = Actual Rain [TP, FN], Row 2 = Actual Sunny [FP, TN]. Rain is class 0 (positive).",
        ]),
        ("B. Dataset & Preprocessing", [
            f"• Camera source: {ds.camera} CCTV feed, Bandung, Indonesia.",
            f"• Text-free crop: keeps x∈[{ds.crop[0]:.2f},{ds.crop[1]:.2f}], y∈[{ds.crop[2]:.2f},{ds.crop[3]:.2f}] of the frame to remove timestamp/label overlays.",
            f"• Balanced dataset: {ds.n_class} rain and {ds.n_class} sunny frames sampled uniformly over each clip.",
            f"• Stratified split (80/20): {n_train // 2} train and {n_test // 2} test frames per class (total {n_train} train, {n_test} test). Fixed random seed 42."
            if ds.split == "stratified"
            else f"• Temporal block split: 10 contiguous blocks per class, blocks 3 and 8 held out as test, 5-frame purge gap on each side ({n_train} train, {n_test} test).",
            "• Tabular features: 24 engineered features extracted per frame from the road ROI and overall scene.",
            "• Vision input (CNN): 64x64 RGB images normalized to [0, 1].",
        ]),
        ("C. Known Limitations & Caveats", list(ds.limitations)),
        ("D. Pipeline Reproduction Commands", [
            f"• Retrain all models: python -m script_tools.weather_ml train --model all --dataset {ds.key}",
            f"• Re-evaluate & export Excel: python -m script_tools.weather_ml evaluate --dataset {ds.key}",
            f"• Retrain individual model: python -m script_tools.weather_ml train --model {{rf,xgb,lgbm,catboost,mlp,tabnet,ftt,cnn}} --dataset {ds.key}",
        ]),
    ]

    curr_row = notes_hdr_row + 2
    for title, items in sections:
        ws.write(curr_row, 1, title, fmt_th_sub)
        curr_row += 1
        for item in items:
            ws.write(curr_row, 1, item, fmt_cell)
            curr_row += 1
        curr_row += 1

    try:
        workbook.close()
    except Exception as e:
        _abort(f"cannot write {out_path.name} ({e}) — if the file is open in Excel, please close it and rerun.")


if __name__ == "__main__":
    import json
    import sys
    from script_tools.weather_ml import DATASETS, MODEL_ORDER

    dataset_key = sys.argv[1] if len(sys.argv) > 1 else "sp_buahbatu"
    ds = DATASETS.get(dataset_key, DATASETS["sp_buahbatu"])

    metrics_files = [ds.models / f"metrics_{m}.json" for m in MODEL_ORDER]
    loaded = []
    for p in metrics_files:
        if p.is_file():
            loaded.append(json.loads(p.read_text()))
    if loaded:
        write_workbook(loaded, ds.xlsx, ds)
        print(f"Exported single vertical sheet workbook to {ds.xlsx}")
    else:
        print(f"No metrics records found in {ds.models} to export.")
