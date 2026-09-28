"""実験記録から学習損失と検証QWKの推移図を生成する。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENTS = Path(__file__).resolve().parent
FIGURES = EXPERIMENTS.parent / "figures"


def read_qwk(method: str) -> list[float]:
    path = EXPERIMENTS / "outputs" / method / "run_summary.json"
    with path.open(encoding="utf-8") as source:
        summary = json.load(source)
    values = summary["validation_qwk_by_epoch"]
    if len(values) != 3:
        raise ValueError(f"{path}: 3エポック分の検証QWKが必要です")
    return values


def read_lora_loss() -> tuple[list[float], list[float]]:
    path = EXPERIMENTS / "outputs" / "lora" / "trainer_state.json"
    with path.open(encoding="utf-8") as source:
        state = json.load(source)
    logs = [entry for entry in state["log_history"] if "loss" in entry]
    return [entry["epoch"] for entry in logs], [entry["loss"] for entry in logs]


def read_full_loss() -> tuple[list[float], list[float]]:
    path = EXPERIMENTS / "outputs" / "full_finetune" / "training_loss_from_colab.json"
    with path.open(encoding="utf-8") as source:
        record = json.load(source)
    losses = record["loss"]
    if len(losses) != 56:
        raise ValueError(f"{path}: 50から2800までの56記録が必要です")
    epochs = [
        step / record["steps_per_epoch"]
        for step in range(record["logging_steps"], 2801, record["logging_steps"])
    ]
    return epochs, losses


def main() -> None:
    lora = read_qwk("lora")
    full = read_qwk("full_finetune")
    lora_epochs, lora_loss = read_lora_loss()
    full_epochs, full_loss = read_full_loss()
    epochs = [1, 2, 3]

    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})

    loss_fig, loss_ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    loss_ax.plot(lora_epochs, lora_loss, color="#0072B2", linewidth=1.8, label="LoRA")
    loss_ax.plot(full_epochs, full_loss, "--", color="#D55E00", linewidth=1.8,
                 label="Full fine-tuning")
    loss_ax.set(xlabel="Epoch", ylabel="Training loss", xticks=epochs,
                xlim=(0, 3.1), ylim=(0, 2.1))
    loss_ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    loss_ax.legend(frameon=False, loc="upper right")
    for suffix in ("svg", "png"):
        loss_fig.savefig(FIGURES / f"training_loss_by_epoch.{suffix}", dpi=240)
    plt.close(loss_fig)

    qwk_fig, qwk_ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    qwk_ax.plot(epochs, lora, "o-", color="#0072B2", linewidth=2,
                markersize=7, label="LoRA")
    qwk_ax.plot(
        epochs,
        full,
        "s--",
        color="#D55E00",
        linewidth=2,
        markersize=7,
        label="Full fine-tuning",
    )
    qwk_ax.set(xlabel="Epoch", ylabel="Validation QWK", xticks=epochs,
               xlim=(0, 3.1), ylim=(0.45, 0.54))
    qwk_ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    qwk_ax.legend(frameon=False, loc="lower right")
    for values, offset in ((lora, 0.003), (full, -0.005)):
        best = max(range(3), key=values.__getitem__)
        qwk_ax.annotate(f"{values[best]:.4f}",
                        (epochs[best], values[best] + offset),
                        ha="center", va="center", fontsize=10)

    for suffix in ("svg", "png"):
        qwk_fig.savefig(FIGURES / f"validation_qwk_by_epoch.{suffix}", dpi=240)
    plt.close(qwk_fig)


if __name__ == "__main__":
    main()
