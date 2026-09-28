"""Try a prompt with the pretrained (non-IT) Gemma 3 270M."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "google/gemma-3-270m"
PROMPT = (
    "次のSNS投稿の書き手本人の感情極性を -2、-1、0、+1、+2 の"
    "いずれか一つだけで答えてください。\n"
    "投稿：今日は友達に会えてすごく嬉しかった。\n"
    "回答："
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default=PROMPT)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    completion = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    print(f"model: {MODEL_NAME}")
    print(f"device: {device}")
    print(f"prompt: {args.prompt}")
    print(f"completion: {completion!r}")


if __name__ == "__main__":
    main()
