
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
    DataCollatorWithPadding,
    set_seed
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# ==========================================
# 1. CAU HINH
# ==========================================

MODEL_NAME = "vinai/phobert-base"

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

SEED = 42

LABEL_NAMES = [
    "negative",
    "neutral",
    "positive"
]

os.makedirs("model_outputs", exist_ok=True)

print("=" * 60)
print("PHOBERT - TEENCODE SENTIMENT")
print("=" * 60)

print(
    "Thiet bi:",
    "CUDA" if torch.cuda.is_available() else "CPU"
)


# ==========================================
# 2. DOC DU LIEU
# ==========================================

DATA_DIR = "data/v2"

train_df = pd.read_csv(
    f"{DATA_DIR}/train_aug.csv"
)

val_df = pd.read_csv(
    f"{DATA_DIR}/val.csv"
)

# Cac tap test can danh gia
test_sets = {
    "test": pd.read_csv(f"{DATA_DIR}/test.csv"),

    "teencode": pd.read_csv(
        f"{DATA_DIR}/test_teencode.csv"
    ),

    "noise_10": pd.read_csv(
        f"{DATA_DIR}/test_noise_10.csv"
    ),

    "noise_30": pd.read_csv(
        f"{DATA_DIR}/test_noise_30.csv"
    ),

    "noise_50": pd.read_csv(
        f"{DATA_DIR}/test_noise_50.csv"
    ),

    "khong_dau": pd.read_csv(
        f"{DATA_DIR}/test_khong_dau.csv"
    )
}


print("\nSO LUONG DU LIEU")

print("Train:", len(train_df))
print("Validation:", len(val_df))

for name, df in test_sets.items():
    print(name, ":", len(df))


# ==========================================
# 3. KIEM TRA DU LIEU
# ==========================================

required_columns = [
    "label_id",
    "text_raw_segmented",
    "text_normalized_segmented"
]

all_datasets = [train_df, val_df]

for df in test_sets.values():
    all_datasets.append(df)


for index, df in enumerate(all_datasets):

    for col in required_columns:

        if col not in df.columns:
            raise ValueError(
                "Thieu cot du lieu: " + col
            )

    if df[required_columns].isnull().any().any():

        raise ValueError(
            "Phat hien du lieu rong!"
        )

    if not df["label_id"].isin([0, 1, 2]).all():

        raise ValueError(
            "label_id chi duoc phep la 0, 1, 2"
        )


print("\nKiem tra du lieu thanh cong!")


# ==========================================
# 4. KHOI TAO TOKENIZER
# ==========================================

print("\nDang tai PhoBERT Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ==========================================
# 5. TAO DATASET
# ==========================================

class SentimentDataset(Dataset):

    def __init__(self, dataframe, text_column):

        self.texts = (
            dataframe[text_column]
            .astype(str)
            .tolist()
        )

        self.labels = (
            dataframe["label_id"]
            .astype(int)
            .tolist()
        )

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


# ==========================================
# 6. CAC CHI SO DANH GIA
# ==========================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


# ==========================================
# 7. LUU KET QUA DU DOAN
# ==========================================

def save_predictions(
    trainer,
    dataframe,
    text_column,
    filename
):

    dataset = SentimentDataset(
        dataframe,
        text_column
    )

    output = trainer.predict(dataset)

    predictions = np.argmax(
        output.predictions,
        axis=-1
    )

    # Tinh xac suat du doan
    logits = torch.tensor(
        output.predictions
    )

    probabilities = torch.softmax(
        logits,
        dim=-1
    ).numpy()

    # Tao bang ket qua
    result = dataframe.copy()

    result["true_label"] = (
        dataframe["label_id"].astype(int)
    )

    result["predicted_label"] = predictions

    result["predicted_name"] = [
        LABEL_NAMES[x] for x in predictions
    ]

    result["confidence"] = (
        probabilities.max(axis=1)
    )

    result.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    # Tinh cac chi so
    accuracy = accuracy_score(
        result["true_label"],
        predictions
    )

    macro_f1 = f1_score(
        result["true_label"],
        predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    # Classification Report
    report = classification_report(
        result["true_label"],
        predictions,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_filename = filename.replace(
        "_predictions.csv",
        "_report.csv"
    )

    report_df.to_csv(
        report_filename,
        encoding="utf-8-sig"
    )

    print("\nTap du lieu:", filename)

    print("Accuracy:", round(accuracy, 4))

    print("Macro F1:", round(macro_f1, 4))

    print(
        "Da luu:",
        filename
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


# ==========================================
# 8. HUAN LUYEN MOT MO HINH
# ==========================================

def train_one_model(
    model_title,
    text_column,
    output_dir
):

    print("\n" + "=" * 60)

    print("BAT DAU HUAN LUYEN:", model_title)

    print("Cot du lieu:", text_column)

    print("=" * 60)

    # Chuan bi du lieu
    train_dataset = SentimentDataset(
        train_df,
        text_column
    )

    val_dataset = SentimentDataset(
        val_df,
        text_column
    )

    # Dam bao cung seed
    set_seed(SEED)

    # Khoi tao PhoBERT
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

    # Cau hinh huan luyen
    args = TrainingArguments(

        output_dir=output_dir,

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        learning_rate=LEARNING_RATE,

        weight_decay=0.01,

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="eval_macro_f1",

        greater_is_better=True,

        save_total_limit=1,

        logging_steps=50,

        report_to="none",

        seed=SEED,

        data_seed=SEED,

        dataloader_num_workers=0
    )

    # Tao Trainer
    trainer = Trainer(

        model=model,

        args=args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )

    # Bat dau huan luyen
    trainer.train()

    # Luu mo hinh tot nhat
    trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)

    # Danh gia validation
    val_metrics = trainer.evaluate()

    print("\nKET QUA VALIDATION:", model_title)

    for key, value in val_metrics.items():

        if isinstance(value, (int, float)):

            print(
                key,
                "=",
                round(value, 4)
            )

    # Danh gia tat ca tap test
    all_results = []

    prefix = output_dir.replace(
        "\\",
        "/"
    ).split("/")[-1]

    for test_name, test_df in test_sets.items():

        filename = (
            f"model_outputs/"
            f"{prefix}_{test_name}_predictions.csv"
        )

        metrics = save_predictions(
            trainer,
            test_df,
            text_column,
            filename
        )

        all_results.append({

            "model": model_title,

            "dataset": test_name,

            "accuracy": metrics["accuracy"],

            "macro_f1": metrics["macro_f1"]

        })

    return {

        "model": model_title,

        "val_accuracy": val_metrics.get(
            "eval_accuracy", 0
        ),

        "val_macro_f1": val_metrics.get(
            "eval_macro_f1", 0
        ),

        "test_results": all_results
    }


# ==========================================
# 9. CHAY HAI THI NGHIEM
# ==========================================

results = []

# ------------------------------------------
# BASELINE
# Su dung du lieu chua chuan hoa Teencode
# ------------------------------------------

baseline_result = train_one_model(

    "Baseline",

    "text_raw_segmented",

    "baseline_model"
)

results.append(baseline_result)


# ------------------------------------------
# PROPOSED
# Su dung du lieu da chuan hoa Teencode
# ------------------------------------------

proposed_result = train_one_model(

    "Proposed",

    "text_normalized_segmented",

    "proposed_model"
)

results.append(proposed_result)


# ==========================================
# 10. LUU BANG SO SANH
# ==========================================

# So sanh Validation
validation_results = []

for result in results:

    validation_results.append({

        "model": result["model"],

        "accuracy": result["val_accuracy"],

        "macro_f1": result["val_macro_f1"]

    })


validation_df = pd.DataFrame(
    validation_results
)

validation_df.to_csv(

    "model_outputs/validation_comparison.csv",

    index=False,

    encoding="utf-8-sig"
)


# So sanh tat ca tap Test
test_results = []

for result in results:

    test_results.extend(
        result["test_results"]
    )


test_results_df = pd.DataFrame(
    test_results
)

test_results_df.to_csv(

    "model_outputs/test_comparison.csv",

    index=False,

    encoding="utf-8-sig"
)


# ==========================================
# 11. IN KET QUA CUOI CUNG
# ==========================================

print("\n" + "=" * 60)

print("SO SANH VALIDATION")

print("=" * 60)

print(
    validation_df.to_string(index=False)
)


print("\n" + "=" * 60)

print("SO SANH TAT CA TAP TEST")

print("=" * 60)

print(
    test_results_df.to_string(index=False)
)


print("\nHOAN TAT HUAN LUYEN!")

print(
    "Ket qua duoc luu trong thu muc model_outputs/"
)
