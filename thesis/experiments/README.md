# 実験コード

## Notebook版

[`load_wrime.ipynb`](./load_wrime.ipynb)をJupyter NotebookまたはGoogle Colabで開き、上から順にセルを実行する。

## WRIME Ver.2の読み込み

Pythonスクリプト版を使用する場合は、リポジトリのルートで次を実行する。

このスクリプトは、`writer.sentiment`をクラスID `0`から`4`へ変換し、Gemma 3 270Mのトークナイザで各投稿をトークンID化する。train・validation・testについてバッチサイズ32の`DataLoader`を作り、最初の学習バッチを表示する。文章の長さは各バッチ内で動的に揃える。

```bash
uv run \
  --python 3.12 \
  --with-requirements thesis/experiments/requirements.txt \
  python thesis/experiments/load_wrime.py
```
