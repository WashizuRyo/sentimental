"""WRIME Ver.2を読み込み、モデルへ入力するバッチを作る。"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Gemma3TextForSequenceClassification,
)


MODEL_NAME = "google/gemma-3-270m"
MAX_LENGTH = 512
BATCH_SIZE = 32
SEED = 42
LABEL_TO_ID = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
ID_TO_LABEL = {
    label_id: str(sentiment)
    for sentiment, label_id in LABEL_TO_ID.items()
}


def calculate_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """データローダー全体に対する正答率を計算する。"""
    correct = 0
    total = 0
    num_batches = len(dataloader)

    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader, start=1):
            labels = batch.pop("labels").to(device)
            inputs = {
                name: tensor.to(device)
                for name, tensor in batch.items()
            }
            predictions = model(**inputs).logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

            if batch_index % 100 == 0 or batch_index == num_batches:
                print(
                    f"  evaluated {batch_index}/{num_batches} batches",
                    flush=True,
                )

    return correct / total


def main() -> None:
    torch.manual_seed(SEED)

    dataset = load_dataset(
        "shunk031/wrime",
        name="ver2",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = Gemma3TextForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id={
            label: label_id
            for label_id, label in ID_TO_LABEL.items()
        },
        problem_type="single_label_classification",
    )

    def tokenize_batch(examples: dict) -> dict:
        encoded = tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
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

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )
    train_generator = torch.Generator().manual_seed(SEED)

    dataloaders = {
        "train": DataLoader(
            tokenized_dataset["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=data_collator,
            generator=train_generator,
        ),
        "validation": DataLoader(
            tokenized_dataset["validation"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=data_collator,
        ),
        "test": DataLoader(
            tokenized_dataset["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=data_collator,
        ),
    }

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    print(f"Evaluation device: {device}")
    for split_name, dataloader in dataloaders.items():
        print(f"Evaluating {split_name}...")
        accuracy = calculate_accuracy(model, dataloader, device)
        print(f"{split_name} accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
