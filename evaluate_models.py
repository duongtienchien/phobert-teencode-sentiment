
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay
)

os.makedirs("model_outputs/evaluation", exist_ok=True)

files = {
    "Baseline_Test": "model_outputs/baseline_model_test_predictions.csv",
    "Proposed_Test": "model_outputs/proposed_model_test_predictions.csv",
    "Baseline_Teencode": "model_outputs/baseline_model_teencode_predictions.csv",
    "Proposed_Teencode": "model_outputs/proposed_model_teencode_predictions.csv"
}

labels = [0, 1, 2]
label_names = ["Negative", "Neutral", "Positive"]

summary = []
reports = []

for name, path in files.items():
    if not os.path.exists(path):
        print("Khong tim thay:", path)
        continue

    df = pd.read_csv(path)
    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=labels,
        zero_division=0
    )

    summary.append({
        "model_dataset": name,
        "samples": len(df),
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    for label_name, metrics in report.items():
        if isinstance(metrics, dict):
            reports.append({
                "model_dataset": name,
                "class": label_name,
                **metrics
            })

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_names
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(name.replace("_", " "))
    fig.tight_layout()

    image_path = f"model_outputs/evaluation/{name}_confusion_matrix.png"
    fig.savefig(image_path, dpi=200)
    plt.close(fig)

    print("\n", name)
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Confusion Matrix:")
    print(cm)
    print("Da luu:", image_path)

pd.DataFrame(summary).to_csv(
    "model_outputs/evaluation/summary.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(reports).to_csv(
    "model_outputs/evaluation/classification_reports.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nDA HOAN TAT!")
print("Bao cao nam trong model_outputs/evaluation/")
