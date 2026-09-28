# ゼロショット評価

`evaluate.py`は指示調整済みの`google/gemma-3-270m-it`を追加学習せずに使い、WRIME Ver.2公式テスト分割の`writer.sentiment`を5クラスに分類する。リポジトリのルートから実行する。

```bash
uv run --python 3.12 --no-project \
  --with-requirements thesis/experiments/requirements.txt \
  python thesis/experiments/zero-shot/evaluate.py
```

最初の10件だけで動作確認する場合は、結果の上書きを避けるため保存先を分ける。

```bash
uv run --python 3.12 --no-project \
  --with-requirements thesis/experiments/requirements.txt \
  python thesis/experiments/zero-shot/evaluate.py \
  --limit 10 --output-dir thesis/experiments/outputs/zero_shot_smoke
```

`outputs/zero_shot/`の`test_results.json`にQWK・Accuracy・MAE、`predictions.jsonl`に各事例の正解と予測、`experiment_environment.json`にモデル・データのリビジョンと推論設定を保存する。GemmaのHugging Face利用条件への同意と認証が必要な環境もある。

推論にはモデル付属のチャットテンプレートを用い、温度・top-pを使わない貪欲デコード（`do_sample=False`）で出力する。生成候補を`-2`、`-1`、`0`、`+1`、`+2`の文字列に制限し、最大生成長をラベルと終端を含む3トークンに固定する。これらの制約は全テスト事例に同じように適用する。入力長の上限はプロンプト全体で512トークンとし、超過時は切り捨てずエラーにする。バッチサイズの初期値は16で、メモリに合わせて`--batch-size`を変更できる。

この結果は、IT版によるラベル生成のゼロショット性能である。base版に分類ヘッドを付けて学習するLoRA・フルFTとは、初期モデルと出力方式が異なる。三者の性能差をWRIMEによる追加学習のみの効果として解釈しない。

2026年9月28日にMac（MPS）で公式テスト2,500件を評価した結果、QWKは0.00038、Accuracyは0.1012、MAEは2.3144だった。予測は2,496件が`-2`に集中した。別途、架空の明るい・悲しい・中立の投稿で診断したところ、生成候補の制約を外しても`-2`を出力し、プロンプトのラベル順を逆にするといずれも先頭の`+2`を出力した。この設定では選択肢の順序が予測を大きく左右する。テスト結果を見ながら設定を選び直す場合は、その結果を当初の単一実験として扱わない。
