# AGENTS.md

このリポジトリは卒論「WRIME Ver.2 を用いた LLM による SNS 投稿の感情分析」の作業用。

## 方針

- 対象タスクは，日本語 SNS 投稿の 5 クラス感情極性分類。
- データセットは `shunk031/wrime` の `ver2`。
- 取得は `datasets.load_dataset("shunk031/wrime", name="ver2")` を前提にする。
- ラベルは `writer.sentiment` を使う。
- ラベル範囲は `-2, -1, 0, +1, +2`。
- モデルは Gemma 3（270M）。

## 比較する手法

- 学習なしのベースモデル
- LoRA ファインチューニング
- フルファインチューニング

比較軸は性能と計算コスト。

## 評価指標

- 主指標: QWK（Quadratic Weighted Kappa）
- 補助指標: Accuracy, MAE

## 実装時の注意

- 分類ヘッドは 5 クラス前提。
- データ分割は WRIME Ver.2 の公式分割に従う。
- 学習用 30,000 件，検証用 2,500 件，テスト用 2,500 件を前提にする。
- 文章や実験設定を変えるときは，`thesis/2_intro.md` の記述と矛盾しないようにする。
- 論文の書式は，まず `thesis/format/thesis_format_manual_20250307.pdf` を優先し，詳細確認が必要な場合は `thesis/format/japanese_psychological_association_writing_manual_2015.pdf` を参照する。
