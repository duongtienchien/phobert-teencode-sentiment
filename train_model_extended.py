
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

# Chay 3 seed cho ca Baseline va Proposed
SEEDS = [42, 123, 456]

LABEL_NAMES = [
    "negative",
    "neutral",
    "positive"
]

DATA_DIR = "data/v2"
OUTPUT_DIR = "model_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PHOBERT - TEENCODE SENTIMENT")
print("=" * 60)

print(
    "Thiet bi:",
    "CUDA" if torch.cuda.is_available() else "CPU"
)

print("Cac seed:", SEEDS)


# ==========================================
# 2. DOC DU LIEU
# ==========================================

train_df = pd.read_csv(
    f"{DATA_DIR}/train_aug.csv"
)

val_df = pd.read_csv(
    f"{DATA_DIR}/val.csv"
)

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

    logits = torch.tensor(
        output.predictions
    )

    probabilities = torch.softmax(
        logits,
        dim=-1
    ).numpy()

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
    print("Da luu:", filename)

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
    output_dir,
    seed
):

    print("\n" + "=" * 60)
    print("BAT DAU HUAN LUYEN:", model_title)
    print("Seed:", seed)
    print("Cot du lieu:", text_column)
    print("Thu muc:", output_dir)
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    train_dataset = SentimentDataset(
        train_df,
        text_column
    )

    val_dataset = SentimentDataset(
        val_df,
        text_column
    )

    # Dat seed truoc khi khoi tao model
    set_seed(seed)

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

        seed=seed,

        data_seed=seed,

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

    # Huan luyen
    trainer.train()

    # Luu checkpoint tot nhat
    trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)

    # Luu trang thai huan luyen de kiem tra seed
    trainer.state.save_to_json(
        os.path.join(
            output_dir,
            "trainer_state.json"
        )
    )

    # Danh gia validation
    val_metrics = trainer.evaluate()

    print("\nKET QUA VALIDATION:", model_title)
    print("Seed:", seed)

    for key, value in val_metrics.items():

        if isinstance(value, (int, float)):

            print(
                key,
                "=",
                round(value, 4)
            )

    # Danh gia cac tap test
    all_results = []

    prefix = os.path.basename(
        os.path.normpath(output_dir)
    )

    for test_name, test_df in test_sets.items():

        filename = os.path.join(
            OUTPUT_DIR,
            f"{prefix}_{test_name}_predictions.csv"
        )

        metrics = save_predictions(
            trainer,
            test_df,
            text_column,
            filename
        )

        all_results.append({

            "seed": seed,

            "model": model_title,

            "dataset": test_name,

            "accuracy": metrics["accuracy"],

            "macro_f1": metrics["macro_f1"]

        })

    return {

        "seed": seed,

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
# 9. CHAY CAC SEED
# ==========================================

all_seed_results = []

for seed in SEEDS:

    print("\n" + "=" * 70)
    print("BAT DAU SEED:", seed)
    print("=" * 70)

    # Thu muc rieng cho tung seed
    baseline_dir = os.path.join(
        OUTPUT_DIR,
        f"baseline_model_seed{seed}"
    )

    proposed_dir = os.path.join(
        OUTPUT_DIR,
        f"proposed_model_seed{seed}"
    )

    # --------------------------------------
    # BASELINE
    # --------------------------------------

    baseline_result = train_one_model(
        "Baseline",
        "text_raw_segmented",
        baseline_dir,
        seed
    )

    # --------------------------------------
    # PROPOSED
    # --------------------------------------

    proposed_result = train_one_model(
        "Proposed",
        "text_normalized_segmented",
        proposed_dir,
        seed
    )

    # --------------------------------------
    # LUU KET QUA VALIDATION
    # --------------------------------------

    validation_results = []

    for result in [baseline_result, proposed_result]:

        validation_results.append({
            "seed": seed,
            "model": result["model"],
            "accuracy": result["val_accuracy"],
            "macro_f1": result["val_macro_f1"]
        })

    validation_df = pd.DataFrame(validation_results)

    validation_filename = os.path.join(
        OUTPUT_DIR,
        f"validation_comparison_seed{seed}.csv"
    )

    validation_df.to_csv(
        validation_filename,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------
    # LUU KET QUA TEST
    # --------------------------------------

    test_results = []

    for result in [baseline_result, proposed_result]:

        test_results.extend(result["test_results"])

    test_results_df = pd.DataFrame(test_results)

    test_filename = os.path.join(
        OUTPUT_DIR,
        f"test_comparison_seed{seed}.csv"
    )

    test_results_df.to_csv(
        test_filename,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------
    # GOM KET QUA
    # --------------------------------------

    all_seed_results.append({
        "seed": seed,
        "validation": validation_results,
        "test": test_results
    })

    print("\nDA HOAN TAT SEED:", seed)

    print("Validation:", validation_filename)
    print("Test:", test_filename)


# ==========================================
# 10. TONG HOP KET QUA CAC SEED
# ==========================================

all_validation_results = []
all_test_results = []

for seed_result in all_seed_results:

    all_validation_results.extend(
        seed_result["validation"]
    )

    all_test_results.extend(
        seed_result["test"]
    )


# Tong hop Validation
all_validation_df = pd.DataFrame(
    all_validation_results
)

all_validation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_comparison_all_seeds.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# Tong hop Test
all_test_df = pd.DataFrame(
    all_test_results
)

all_test_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_comparison_all_seeds.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 11. IN KET QUA CUOI CUNG
# ==========================================

print("\n" + "=" * 70)
print("TONG HOP KET QUA VALIDATION")
print("=" * 70)

print(
    all_validation_df.to_string(index=False)
)

print("\n" + "=" * 70)
print("TONG HOP KET QUA TEST")
print("=" * 70)

print(
    all_test_df.to_string(index=False)
)

print("\nHOAN TAT TAT CA CAC SEED!")

print(
    "Ket qua duoc luu trong thu muc model_outputs/"
)
