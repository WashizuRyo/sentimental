"""WRIME Ver.2でGemma 3 270MのLoRA文章分類モデルを学習する。"""

import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Gemma3TextForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


# フルファインチューニングと同じ初期モデル・データ・評価条件を用いる。
MODEL_NAME = "google/gemma-3-270m"
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
DATASET_NAME = "shunk031/wrime"
DATASET_REVISION = "3fb7212c389d7818b8e6179e2cdac762f2e081d9"
OUTPUT_DIR = Path("thesis/experiments/outputs/lora")
MAX_LENGTH = 512
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
SEED = 42
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]
CLASSIFIER_MODULE = "score"
LABEL_TO_ID = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
ID_TO_LABEL = {label_id: str(label) for label, label_id in LABEL_TO_ID.items()}


def save_experiment_environment() -> Path:
    """依存関係・GPU・入力とLoRA設定を記録する。"""
    device_index = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    environment = {
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "datasets": version("datasets"),
        "accelerate": version("accelerate"),
        "peft": version("peft"),
        "scikit_learn": version("scikit-learn"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(device_index),
        "gpu_total_memory_gb": float(device_properties.total_memory / 1024**3),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_name": MODEL_NAME,
        "tokenizer_revision": MODEL_REVISION,
        "dataset_name": DATASET_NAME,
        "dataset_config": "ver2",
        "dataset_revision": DATASET_REVISION,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "modules_to_save": [CLASSIFIER_MODULE],
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "seed": SEED,
    }
    path = OUTPUT_DIR / "experiment_environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
    """順序尺度上でQWK・Accuracy・MAEを計算する。"""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "qwk": float(cohen_kappa_score(labels, predictions, labels=list(range(5)), weights="quadratic")),
        "accuracy": float(accuracy_score(labels, predictions)),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("最大GPUメモリの計測にはCUDA環境が必要です。")

    set_seed(SEED)
    dataset = load_dataset(
        DATASET_NAME, name="ver2", revision=DATASET_REVISION, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

    def tokenize_batch(examples: dict) -> dict:
        encoded = tokenizer(examples["sentence"], truncation=True, max_length=MAX_LENGTH)
        encoded["labels"] = [LABEL_TO_ID[writer["sentiment"]] for writer in examples["writer"]]
        return encoded

    tokenized_dataset = dataset.map(
        tokenize_batch, batched=True, remove_columns=dataset["train"].column_names
    )

    base_model = Gemma3TextForSequenceClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id={label: label_id for label_id, label in ID_TO_LABEL.items()},
        problem_type="single_label_classification",
    )
    base_model.config.use_cache = False
    if not hasattr(base_model, CLASSIFIER_MODULE):
        raise RuntimeError(f"分類ヘッド {CLASSIFIER_MODULE} が見つかりません。")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        modules_to_save=[CLASSIFIER_MODULE],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or not any("lora_" in name for name in trainable_names):
        raise RuntimeError("LoRAパラメータが学習対象になっていません。")
    if not any(CLASSIFIER_MODULE in name for name in trainable_names):
        raise RuntimeError("分類ヘッドが学習対象になっていません。")
    if any("lora_" not in name and CLASSIFIER_MODULE not in name for name in trainable_names):
        raise RuntimeError("モデル本体に意図しない学習対象パラメータがあります。")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        weight_decay=0.0,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="qwk",
        greater_is_better=True,
        save_total_limit=3,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    environment_path = save_experiment_environment()
    print(f"Experiment environment saved to: {environment_path}")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    torch.cuda.synchronize()
    train_result.metrics["max_gpu_memory_allocated_gb"] = float(torch.cuda.max_memory_allocated() / 1024**3)
    train_result.metrics["max_gpu_memory_reserved_gb"] = float(torch.cuda.max_memory_reserved() / 1024**3)
    train_result.metrics["total_parameters"] = total_parameters
    train_result.metrics["trainable_parameters"] = trainable_parameters
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # 最良チェックポイントのアダプタには、学習した分類ヘッドも含まれる。
    best_adapter_dir = OUTPUT_DIR / "best_adapter"
    trainer.save_model(str(best_adapter_dir))

    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["test"], metric_key_prefix="test"
    )
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    # 論文で予定する推論用の統合モデルも保存する。
    merged_model = trainer.model.merge_and_unload()
    merged_dir = OUTPUT_DIR / "merged_model"
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Best validation QWK: {trainer.state.best_metric}")
    print(f"Best adapter saved to: {best_adapter_dir}")
    print(f"Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
