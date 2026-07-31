# 実験コード

## WRIME Ver.2の読み込み

Pythonスクリプト版を使用する場合は、リポジトリのルートで次を実行する。

このスクリプトは、`writer.sentiment`をクラスID `0`から`4`へ変換し、Gemma 3 270Mの未学習分類ヘッドでtrain・validation・testのAccuracyを算出する。

```bash
uv run \
  --python 3.12 \
  --with-requirements thesis/experiments/requirements.txt \
  python thesis/experiments/load_wrime.py
```

## 最大入力長による切り捨ての確認

[`check_truncation.py`](./check_truncation.py)は、フルファインチューニングと同じGemmaトークナイザを用いて、最大入力長512トークンを超える投稿数、割合、および各分割の最大トークン長を集計する。

```bash
uv run \
  --python 3.12 \
  --with datasets==2.21.0 \
  --with transformers==5.14.1 \
  python thesis/experiments/check_truncation.py
```

集計結果は`thesis/experiments/outputs/truncation_stats.json`へ保存される。固定済みのモデルを初めて取得する環境では、事前にHugging Face上でGemmaの利用条件へ同意し、認証する必要がある。

## フルファインチューニング

[`full_finetune.py`](./full_finetune.py)は、Gemma 3 270Mの全パラメータと5クラス分類ヘッドをWRIME Ver.2の学習分割で3エポック学習する。各エポック後に検証分割でQWK、Accuracy、MAEを計算し、検証QWKが最も高いチェックポイントを選択する。学習終了後に限り、選択したモデルをテスト分割で評価する。

学習開始前に、ソフトウェア、CUDA、cuDNN、GPU、入力リビジョン、実行日時を`thesis/experiments/outputs/full_finetune/experiment_environment.json`へ保存する。

実験を開始する場合は、NVIDIA T4 GPUを有効にしたGoogle ColaboratoryなどのCUDA環境で、リポジトリのルートから次を実行する。

```bash
uv run \
  --python 3.12 \
  --with-requirements thesis/experiments/requirements.txt \
  python thesis/experiments/full_finetune.py
```
