"""WRIME Ver.2で最大入力長を超える投稿数を確認する。"""

import json
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


MODEL_NAME = "google/gemma-3-270m"
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
DATASET_NAME = "shunk031/wrime"
DATASET_REVISION = "3fb7212c389d7818b8e6179e2cdac762f2e081d9"
MAX_LENGTH = 512
TOKENIZE_BATCH_SIZE = 1_000
OUTPUT_PATH = Path(
    "thesis/experiments/outputs/truncation_stats.json"
)


def calculate_split_stats(
    split: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, int | float]:
    """一つの分割について、トークン長と切り捨て対象件数を集計する。"""
    total = len(split)
    over_max_length = 0
    max_token_length = 0

    for start in range(0, total, TOKENIZE_BATCH_SIZE):
        sentences = split[start : start + TOKENIZE_BATCH_SIZE][
            "sentence"
        ]
        encoded = tokenizer(
            sentences,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_length=True,
        )
        lengths = encoded["length"]
        over_max_length += sum(
            length > MAX_LENGTH
            for length in lengths
        )
        max_token_length = max(max_token_length, max(lengths))

    return {
        "total": total,
        "over_max_length": over_max_length,
        "over_max_length_ratio": over_max_length / total,
        "max_token_length": max_token_length,
    }


def main() -> None:
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

    split_stats = {
        split_name: calculate_split_stats(split, tokenizer)
        for split_name, split in dataset.items()
    }
    total = sum(
        stats["total"]
        for stats in split_stats.values()
    )
    total_over_max_length = sum(
        stats["over_max_length"]
        for stats in split_stats.values()
    )
    results = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dataset_name": DATASET_NAME,
        "dataset_revision": DATASET_REVISION,
        "max_length": MAX_LENGTH,
        "splits": split_stats,
        "overall": {
            "total": total,
            "over_max_length": total_over_max_length,
            "over_max_length_ratio": (
                total_over_max_length / total
            ),
            "max_token_length": max(
                stats["max_token_length"]
                for stats in split_stats.values()
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for split_name, stats in split_stats.items():
        print(
            f"{split_name}: "
            f"{stats['over_max_length']}/{stats['total']} "
            f"({stats['over_max_length_ratio']:.2%}), "
            f"max={stats['max_token_length']} tokens"
        )
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
