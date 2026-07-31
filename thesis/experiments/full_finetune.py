"""Gemma 3 270MをWRIME Ver.2でフルファインチューニングする。"""

import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
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


MODEL_NAME = "google/gemma-3-270m"
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
DATASET_NAME = "shunk031/wrime"
DATASET_REVISION = "3fb7212c389d7818b8e6179e2cdac762f2e081d9"
OUTPUT_DIR = Path("thesis/experiments/outputs/full_finetune")
MAX_LENGTH = 512
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
SEED = 42
# 感情極性を、分類ヘッドが扱う連続したクラスIDへ変換する。
LABEL_TO_ID = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
ID_TO_LABEL = {
    label_id: str(sentiment)
    for sentiment, label_id in LABEL_TO_ID.items()
}


def save_experiment_environment() -> Path:
    """実験環境と入力リビジョンをJSONへ保存する。"""
    device_index = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)
    environment = {
        "executed_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "datasets": version("datasets"),
        "accelerate": version("accelerate"),
        "scikit_learn": version("scikit-learn"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(device_index),
        "gpu_total_memory_gb": float(
            device_properties.total_memory / 1024**3
        ),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_name": MODEL_NAME,
        "tokenizer_revision": MODEL_REVISION,
        "dataset_name": DATASET_NAME,
        "dataset_config": "ver2",
        "dataset_revision": DATASET_REVISION,
    }

    environment_path = OUTPUT_DIR / "experiment_environment.json"
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return environment_path


def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
    """QWK、Accuracy、MAEをクラスID 0から4の順序尺度上で計算する。"""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "qwk": float(
            cohen_kappa_score(
                labels,
                predictions,
                labels=list(range(len(LABEL_TO_ID))),
                weights="quadratic",
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "mae": float(mean_absolute_error(labels, predictions)),
    }


def main() -> None:
    # 分類ヘッドの初期値やデータの並び順を再現可能にする。
    set_seed(SEED)

    # WRIME Ver.2の公式train・validation・test分割を読み込む。
    dataset = load_dataset(
        DATASET_NAME,
        name="ver2",
        revision=DATASET_REVISION,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    def tokenize_batch(examples: dict) -> dict:
        encoded = tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # 書き手本人の感情極性を正解ラベルとして使用する。
        encoded["labels"] = [
            LABEL_TO_ID[writer["sentiment"]]
            for writer in examples["writer"]
        ]
        return encoded

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Gemma 3本体へ、出力数5の文章分類ヘッドを追加する。
    model = Gemma3TextForSequenceClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id={
            label: label_id
            for label_id, label in ID_TO_LABEL.items()
        },
        problem_type="single_label_classification",
    )
    # 学習時には生成用のKVキャッシュが不要なため無効化する。
    model.config.use_cache = False

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    # 一部が意図せず凍結されていないことを学習前に検証する。
    if trainable_parameters != total_parameters:
        raise RuntimeError(
            "フルファインチューニングでは全パラメータが"
            "学習可能である必要があります。"
        )

    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    # 各エポック後に検証し、検証QWKが最大のモデルを選択する。
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

    # ミニバッチごとの長さは、最長の文章に合わせて動的に揃える。
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # trainのみで更新し、validationはモデル選択にだけ使用する。
    if not torch.cuda.is_available():
        raise RuntimeError(
            "最大GPUメモリの計測にはCUDA環境が必要です。"
        )
    environment_path = save_experiment_environment()
    print(f"Experiment environment saved to: {environment_path}")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    train_result = trainer.train()
    torch.cuda.synchronize()
    train_result.metrics["max_gpu_memory_allocated_gb"] = float(
        torch.cuda.max_memory_allocated() / 1024**3
    )
    train_result.metrics["max_gpu_memory_reserved_gb"] = float(
        torch.cuda.max_memory_reserved() / 1024**3
    )
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # train()終了時点では、最高QWKのチェックポイントが復元されている。
    best_model_dir = OUTPUT_DIR / "best_model"
    trainer.save_model(str(best_model_dir))

    # testは最良モデルの決定後に一度だけ最終評価へ使用する。
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["test"],
        metric_key_prefix="test",
    )
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Best validation QWK: {trainer.state.best_metric}")
    print(f"Best model saved to: {best_model_dir}")


if __name__ == "__main__":
    main()
