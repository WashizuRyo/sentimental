# LoRAファインチューニング

`train.py`はWRIME Ver.2の`writer.sentiment`を使い、Gemma 3 270Mに5クラス分類ヘッドを追加してLoRA学習する。Attentionの`q_proj`と`v_proj`にLoRAを適用し、分類ヘッド`score`も学習・保存する。ランク8、スケーリング係数32、ドロップアウト0.1、学習率2e-4とし、それ以外の主な学習・評価条件は`../full_finetune.py`に合わせた。

CUDA GPUとGemmaのHugging Face利用許諾・認証を用意し、リポジトリのルートから実行する。

```bash
uv run \
  --python 3.12 \
  --with-requirements thesis/experiments/requirements.txt \
  python thesis/experiments/lora/train.py
```

結果は`thesis/experiments/outputs/lora/`に保存される。`train_results.json`には総パラメータ数、学習可能パラメータ数、Trainerの`train_runtime`、ピークGPUメモリなどを、`test_results.json`には最良検証QWKのチェックポイントによる最終評価を記録する。`best_adapter/`にはLoRA重みと分類ヘッド、`merged_model/`には統合済みモデルを保存する。テスト評価はモデル選択後に一度だけ行う。

既存のフルファインチューニングコードに合わせて、初期モデルは`google/gemma-3-270m`（事前学習済み）を使用する。論文の`3.3 使用モデル`には`google/gemma-3-270m-it`（指示調整済み）と書かれているため、結果を論文に反映する前にモデル記述を実際の実験条件へ合わせること。なお`train_runtime`にはチェックポイント保存時間が含まれ、論文の「保存時間を含めない」という定義とは一致しない。この点もフル学習との比較前に整理する。
