"""
evaluation/run_all.py
---------------------
Nhanh feat/evaluation 
Chay 1 lenh de tao toan bo ket qua danh gia:
  python evaluation/run_all.py

Dau ra:
  evaluation/01_metrics/            -- F1, Confusion Matrix 6 tap, bieu do nhieu
  evaluation/02_error_analysis/     -- Error analysis chi tiet dung replacements
  evaluation/03_inference_pipeline/ -- Demo pipeline
"""

import os, sys, subprocess, time

BASE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    (fr"{BASE}\update_after_merge.py",                      "[1] Tinh Metrics toan dien tren 6 tap test & Ve bieu do"),
    (fr"{BASE}\02_error_analysis\error_analysis.py",        "[2] Error Analysis (replacements, phan loai loi mia mai/phu dinh)"),
    (fr"{BASE}\03_inference_pipeline\inference_pipeline.py","[3] Inference Pipeline Demo"),
]

print("\n" + "=" * 65)
print("  EVALUATION PIPELINE — feat/evaluation ")
print("=" * 65)

start_total = time.time()
all_ok = True

for script_path, label in SCRIPTS:
    print(f"\n>>> {label}")
    print(f"    Script: {os.path.basename(script_path)}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"    OK ({elapsed:.1f}s)")
    else:
        print(f"    FAILED (code={result.returncode})")
        all_ok = False

elapsed_total = time.time() - start_total
print("\n" + "=" * 65)
if all_ok:
    print(f"  HOAN THANH! ({elapsed_total:.1f}s)")
    print()
    print("  Ket qua:")
    print("    evaluation/01_metrics/")
    print("      - metrics_all_sets.csv           : F1, Accuracy, Precision, Recall day du 6 tap test")
    print("      - main_results_table.png         : Bang tong hop ket qua chinh Baseline vs Proposed")
    print("      - f1_noise_curve.png             : Duong cong F1 theo muc nhieu (0% -> 10% -> 30% -> 50%)")
    print("      - confusion_matrices_Baseline.png: 6 Confusion Matrix cua Baseline")
    print("      - confusion_matrices_Proposed.png: 6 Confusion Matrix cua Proposed")
    print()
    print("    evaluation/02_error_analysis/")
    print("      - error_summary.csv              : Tom tat loi theo model & tap")
    print("      - error_detail_all.csv           : Chi tiet tung cau sai")
    print("      - error_list_for_ni1.csv         : Danh sach cau sai kem replacements GUI NI 1")
    print("      - error_rate_comparison.png      : So sanh ti le loi theo tap test")
    print("      - error_analysis_charts.png      : Bieu do phan tich loai loi")
    print("      - confidence_distribution.png    : Phan bo do tu tin")
    print()
    print("    evaluation/03_inference_pipeline/")
    print("      - pipeline_demo_output.csv       : Demo cau tho -> normalizer -> pyvi -> PhoBERT")
else:
    print("  CO LOI! Kiem tra output o tren.")
print("=" * 65)
