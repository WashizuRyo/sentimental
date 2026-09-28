"""Gemma 3 270M ITのゼロショット感情極性分類をWRIME Ver.2で評価する。"""

import argparse
import json
import platform
import time
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


MODEL_NAME = "google/gemma-3-270m-it"
MODEL_REVISION = "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
DATASET_NAME = "shunk031/wrime"
DATASET_REVISION = "3fb7212c389d7818b8e6179e2cdac762f2e081d9"
OUTPUT_DIR = Path("thesis/experiments/outputs/zero_shot")
LABELS = ("-2", "-1", "0", "+1", "+2")
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 3  # 2トークンのラベルと終端トークン
BATCH_SIZE = 16
SEED = 42
INSTRUCTION = (
    "次のSNS投稿に表れている，書き手本人の感情極性を分類してください。\n"
    "ラベルは，-2（強いネガティブ），-1（ネガティブ），0（中立），"
    "+1（ポジティブ），+2（強いポジティブ）の5段階です。\n"
    "回答は「-2」「-1」「0」「+1」「+2」のいずれか一つだけを出力してください。\n\n"
    "SNS投稿：{sentence}\n回答："
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, help="動作確認用。テスト分割の先頭N件だけ評価する")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size と --limit は正の整数にしてください")
    return args


def make_allowed_tokens_fn(label_token_ids: tuple[tuple[int, ...], ...], end_id: int, prompt_length: int):
    """既に生成した接頭辞に続けられるラベルのトークンだけを返す。"""

    def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        generated = tuple(input_ids[prompt_length:].tolist())
        allowed = set()
        for label_ids in label_token_ids:
            if generated == label_ids:
                allowed.add(end_id)
            elif label_ids[: len(generated)] == generated:
                allowed.add(label_ids[len(generated)])
        if not allowed:
            raise RuntimeError(f"ラベル制約から外れたトークン列: {generated}")
        return sorted(allowed)

    return allowed_tokens


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.float16 if device == "cuda" else torch.float32

    # 学習・検証分割は読み込まず、公式テスト分割のみを評価する。
    test = load_dataset(
        DATASET_NAME,
        name="ver2",
        split="test",
        revision=DATASET_REVISION,
        trust_remote_code=True,
    )
    if args.limit is not None:
        test = test.select(range(min(args.limit, len(test))))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=dtype
    ).to(device).eval()

    label_token_ids = tuple(
        tuple(tokenizer.encode(label, add_special_tokens=False)) for label in LABELS
    )
    end_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if any(not ids for ids in label_token_ids) or end_id == tokenizer.unk_token_id:
        raise RuntimeError("ラベルまたは終端トークンをトークン化できません")
    if max(map(len, label_token_ids)) + 1 > MAX_NEW_TOKENS:
        raise RuntimeError("MAX_NEW_TOKENS がラベルと終端トークンに不足しています")

    predictions = []
    correct_labels = []
    records = []
    started = time.perf_counter()
    for start in range(0, len(test), args.batch_size):
        batch = test[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": INSTRUCTION.format(sentence=sentence)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for sentence in batch["sentence"]
        ]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        if inputs["input_ids"].shape[1] > MAX_INPUT_LENGTH:
            raise ValueError("プロンプトが最大入力長512トークンを超えました")
        inputs = inputs.to(device)
        prompt_length = inputs["input_ids"].shape[1]
        allowed_tokens_fn = make_allowed_tokens_fn(label_token_ids, end_id, prompt_length)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                eos_token_id=end_id,
                pad_token_id=tokenizer.pad_token_id,
                prefix_allowed_tokens_fn=allowed_tokens_fn,
            )

        for offset, (tokens, writer) in enumerate(zip(output, batch["writer"], strict=True)):
            generated = tokenizer.decode(tokens[prompt_length:], skip_special_tokens=True).strip()
            if generated not in LABELS:
                raise RuntimeError(f"予期しない出力: {generated!r}")
            gold = int(writer["sentiment"])
            predicted = int(generated)
            correct_labels.append(gold)
            predictions.append(predicted)
            records.append({
                "test_index": start + offset,
                "gold": gold,
                "prediction": predicted,
                "generated_text": generated,
            })
        print(f"Evaluated {len(records)}/{len(test)}", flush=True)

    elapsed = time.perf_counter() - started
    results = {
        "test_qwk": float(cohen_kappa_score(
            correct_labels, predictions, labels=[-2, -1, 0, 1, 2], weights="quadratic"
        )),
        "test_accuracy": float(accuracy_score(correct_labels, predictions)),
        "test_mae": float(mean_absolute_error(correct_labels, predictions)),
        "test_samples": len(test),
        "test_total_samples": 2500,
        "test_is_full_split": args.limit is None or len(test) == 2500,
        "invalid_outputs": 0,
        "prediction_counts": {
            label: Counter(predictions)[int(label)] for label in LABELS
        },
        "inference_seconds": elapsed,
    }
    environment = {
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "datasets": version("datasets"),
        "scikit_learn": version("scikit-learn"),
        "device": device,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
        "model_dtype": str(dtype),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dataset_name": DATASET_NAME,
        "dataset_config": "ver2",
        "dataset_revision": DATASET_REVISION,
        "split": "test",
        "prompt_template": INSTRUCTION,
        "labels": LABELS,
        "label_token_ids": label_token_ids,
        "chat_template": tokenizer.chat_template,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": args.batch_size,
        "decoding": "greedy, constrained to the five label strings",
        "temperature": None,
        "top_p": None,
        "seed": SEED,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "experiment_environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
