
import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ==============================
# 1. CAU HINH
# ==============================

MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3

LABEL_NAMES = ["negative", "neutral", "positive"]

os.makedirs("model_outputs", exist_ok=True)

print("Thiet bi:", "CUDA" if torch.cuda.is_available() else "CPU")


# ==============================
# 2. DOC DU LIEU
# ==============================

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")
teencode_df = pd.read_csv("test_teencode.csv")

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
print("Test Teencode:", len(teencode_df))

# Kiem tra cac cot bat buoc
required_columns = [
    "label_id",
    "text_raw_segmented",
    "text_normalized_segmented"
]

for df in [train_df, val_df, test_df, teencode_df]:
    for col in required_columns:
        if col not in df.columns:
            raise ValueError("Thieu cot du lieu: " + col)

    if df[required_columns].isnull().any().any():
        raise ValueError("Phat hien du lieu rong trong cac cot bat buoc.")

    if not df["label_id"].isin([0, 1, 2]).all():
        raise ValueError("label_id chi duoc phep la 0, 1, 2.")

print("Kiem tra du lieu thanh cong!")


# ==============================
# 3. KHOI TAO TOKENIZER
# ==============================

print("Dang tai PhoBERT Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ==============================
# 4. TAO DATASET
# ==============================

class SentimentDataset(Dataset):
    def __init__(self, dataframe, text_column):
        self.texts = dataframe[text_column].astype(str).tolist()
        self.labels = dataframe["label_id"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoded = tokenizer(
            self.texts[index],
            truncation=True,
            max_length=MAX_LENGTH
        )

        encoded["labels"] = self.labels[index]

        return encoded


# ==============================
# 5. TINH CHI SO DANH GIA
# ==============================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
    }


# ==============================
# 6. DU DOAN VA LUU KET QUA
# ==============================

def save_predictions(trainer, dataframe, text_column, filename):
    dataset = SentimentDataset(dataframe, text_column)

    output = trainer.predict(dataset)

    predictions = np.argmax(output.predictions, axis=-1)

    # Xac suat cua nhan duoc du doan
    logits = torch.tensor(output.predictions)
    probabilities = torch.softmax(logits, dim=-1).numpy()

    result = dataframe.copy()
    result["true_label"] = dataframe["label_id"].astype(int)
    result["predicted_label"] = predictions
    result["predicted_name"] = [
        LABEL_NAMES[x] for x in predictions
    ]
    result["confidence"] = probabilities.max(axis=1)

    result.to_csv(filename, index=False, encoding="utf-8-sig")

    print("Da luu:", filename)


# ==============================
# 7. HUAN LUYEN MOT MO HINH
# ==============================

def train_one_model(model_title, text_column, output_dir):
    print("\n" + "=" * 55)
    print("BAT DAU:", model_title)
    print("Cot du lieu:", text_column)
    print("=" * 55)

    train_dataset = SentimentDataset(train_df, text_column)
    val_dataset = SentimentDataset(val_df, text_column)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={
            0: "negative",
            1: "neutral",
            2: "positive"
        },
        label2id={
            "negative": 0,
            "neutral": 1,
            "positive": 2
        }
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=42,
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Luu checkpoint tot nhat
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Danh gia tren validation
    metrics = trainer.evaluate()

    print("\nKet qua validation:", model_title)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(key, "=", round(value, 4))

    # Du doan tren hai tap test
    prefix = output_dir.replace("\\", "/").split("/")[-1]

    save_predictions(
        trainer,
        test_df,
        text_column,
        f"model_outputs/{prefix}_test_predictions.csv"
    )

    save_predictions(
        trainer,
        teencode_df,
        text_column,
        f"model_outputs/{prefix}_teencode_predictions.csv"
    )

    return {
        "model": model_title,
        "accuracy": metrics.get("eval_accuracy", 0),
        "macro_f1": metrics.get("eval_macro_f1", 0)
    }


# ==============================
# 8. CHAY HAI THI NGHIEM
# ==============================

results = []

# Mo hinh Baseline: giu nguyen teencode
results.append(
    train_one_model(
        "Baseline",
        "text_raw_segmented",
        "baseline_model"
    )
)

# Mo hinh de xuat: chuan hoa teencode
results.append(
    train_one_model(
        "Proposed",
        "text_normalized_segmented",
        "proposed_model"
    )
)

# Luu bang so sanh validation
results_df = pd.DataFrame(results)
results_df.to_csv(
    "model_outputs/validation_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 55)
print("KET QUA SO SANH HAI MO HINH")
print("=" * 55)
print(results_df.to_string(index=False))
print("\nHOAN TAT HUAN LUYEN!")
