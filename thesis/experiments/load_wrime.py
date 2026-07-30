"""WRIME Ver.2を読み込み、モデルへ入力するバッチを作る。"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


MODEL_NAME = "google/gemma-3-270m-it"
MAX_LENGTH = 512
BATCH_SIZE = 32
SEED = 42
LABEL_TO_ID = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
ID_TO_LABEL = {
    label_id: str(sentiment)
    for sentiment, label_id in LABEL_TO_ID.items()
}


def main() -> None:
    dataset = load_dataset(
        "shunk031/wrime",
        name="ver2",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
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

    first_batch = next(iter(dataloaders["train"]))
    print(type(model).__name__)
    print({name: tuple(tensor.shape) for name, tensor in first_batch.items()})
    print(first_batch["labels"])


if __name__ == "__main__":
    main()
