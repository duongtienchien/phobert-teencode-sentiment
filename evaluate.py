import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Tạo thư mục lưu kết quả nếu chưa có
os.makedirs("evaluation_results", exist_ok=True)

# 1. Biểu đồ phân bố nhãn
def plot_label_distribution():
    print("--- 1. Đang vẽ biểu đồ phân bố dữ liệu ---")
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    test_df = pd.read_csv("test.csv")
    
    train_df['split'] = 'Train'
    val_df['split'] = 'Validation'
    test_df['split'] = 'Test'
    
    combined_df = pd.concat([train_df, val_df, test_df])
    
    plt.figure(figsize=(8, 5))
    sns.countplot(data=combined_df, x='label_id', hue='split', palette='Set2')
    plt.title('Phan bo nhan tren cac tap du lieu (0: Tieu cuc, 1: Trung tinh, 2: Tich cuc)')
    plt.xlabel('Nhan (label_id)')
    plt.ylabel('So luong binh luan')
    plt.legend(title='Tap du lieu')
    plt.tight_layout()
    plt.savefig("evaluation_results/label_distribution.png")
    plt.close()
    print("-> Da luu bieu do phan bo tai: evaluation_results/label_distribution.png\n")

# 2. Đánh giá Macro F1, Confusion Matrix, Error Analysis
def evaluate_predictions(pred_csv_path, output_prefix="baseline"):
    if not os.path.exists(pred_csv_path):
        print(f"Chua tim thay file {pred_csv_path}. Can Ni 2 (Hieu) xuat file du doan truoc!")
        return

    print(f"--- Dang danh gia file: {pred_csv_path} ---")
    df = pd.read_csv(pred_csv_path)
    
    y_true = df['label_id']
    y_pred = df['pred_label']
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"==> Macro F1-score ({output_prefix}): {macro_f1:.4f}")
    
    report = classification_report(y_true, y_pred, target_names=['Tieu cuc (0)', 'Trung tinh (1)', 'Tich cuc (2)'])
    print("Bao cao chi tiet:")
    print(report)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Tieu cuc', 'Trung tinh', 'Tich cuc'],
                yticklabels=['Tieu cuc', 'Trung tinh', 'Tich cuc'])
    plt.title(f'Confusion Matrix ({output_prefix})')
    plt.xlabel('Du doan (Predicted)')
    plt.ylabel('Thuc te (Actual)')
    plt.tight_layout()
    plt.savefig(f"evaluation_results/confusion_matrix_{output_prefix}.png")
    plt.close()
    
    # Xuất file phân tích lỗi
    errors_df = df[df['label_id'] != df['pred_label']]
    errors_df.to_csv(f"evaluation_results/error_analysis_{output_prefix}.csv", index=False)
    print(f"-> Da luu file phan tich loi tai: evaluation_results/error_analysis_{output_prefix}.csv\n")

if __name__ == "__main__":
    plot_label_distribution()