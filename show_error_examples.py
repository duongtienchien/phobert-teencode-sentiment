
import pandas as pd

df = pd.read_csv(
    "model_outputs/error_analysis/all_comparisons.csv"
)

groups = [
    ("Proposed sửa đúng", "Proposed only correct"),
    ("Proposed sai thêm", "Baseline only correct"),
    ("Cả hai mô hình đều sai", "Both wrong")
]

for title, group_name in groups:

    print("\n" + "=" * 65)
    print(title.upper())
    print("=" * 65)

    data = df[df["error_type"] == group_name]

    print("Tong so:", len(data))

    for _, row in data.head(10).iterrows():

        print("\nID:", row["id"])
        print("Cau:", row["comment"])
        print("Nhan that:", row["true_label"])
        print(
            "Baseline:",
            row["baseline_pred"],
            "| Confidence:",
            round(row["baseline_confidence"], 4)
        )
        print(
            "Proposed:",
            row["proposed_pred"],
            "| Confidence:",
            round(row["proposed_confidence"], 4)
        )
