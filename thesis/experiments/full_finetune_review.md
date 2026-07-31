# `full_finetune.py` の論文整合性・実装レビュー

## 1. レビュー範囲

`full_finetune.py`について，既存の論文本文，WRIME Ver.2の公開データ，Hugging Face Datasets 2.21.0，Transformers 5.14.1，およびPyTorchの公式仕様と照合した。

`full_finetune.py`が`google/gemma-3-270m`を使用する一方，論文本文が`google/gemma-3-270m-it`を使用すると記載している点は，実験上の意図的な変更であるため，本レビューの指摘対象から除外する。

実際のフルファインチューニングは実行せず，コードの静的検証，公開されているWRIME Ver.2原データの確認，および各ライブラリの公式仕様との照合を行った。

## 2. 結論

データ分割，予測対象，ラベル変換，分類ヘッド，評価指標，モデル選択，およびテスト手順は，論文の中心的な実験設定とおおむね一致している。

一方，次の点は実験を開始する前に修正する必要がある。

1. **対応済み:** 新規環境でWRIME Ver.2の読み込みが確認待ちまたはエラーになる問題。
2. 論文で定義した条件どおりの総学習時間を測定できない。
3. **対応済み:** 論文で評価対象としている最大GPUメモリ使用量の計測。
4. **一部対応済み:** モデル・データ・依存パッケージの固定と実行環境の自動記録は対応したが，オプティマイザ設定の明示が残っている。

また，乱数シード1種類のみの比較と，ハイパーパラメータを各手法1設定だけ使用する比較は，コードと論文で一致しているものの，研究設計としては改善が望ましい。

## 3. 優先して修正すべき点

### 3.1 WRIME Ver.2の読み込みが新規環境で止まる可能性がある

**対応済み（2026年7月31日）:** `full_finetune.py`にデータセットリポジトリの完全なコミットSHAと`trust_remote_code=True`を追加した。

レビュー時の該当箇所は`full_finetune.py:59`であり，対応後の読み込み処理は`full_finetune.py:61`から始まる。

```python
dataset = load_dataset("shunk031/wrime", name="ver2")
```

`shunk031/wrime`は，リポジトリ内のデータセットスクリプト`wrime.py`を実行してデータを読み込む形式である。使用する`datasets==2.21.0`では，キャッシュが存在しない場合，リモートコードの実行を許可する確認が表示される。非対話環境では，確認に応答できずエラーになる可能性がある。

同じディレクトリの`load_wrime.py`では，既に`trust_remote_code=True`が指定されているため，二つの実装も一致していない。

修正時は，リモートコードの実行を許可するだけでなく，再現性と安全性のためデータセットのコミットも固定する。

```python
dataset = load_dataset(
    "shunk031/wrime",
    name="ver2",
    revision="<WRIMEデータセットのコミットSHA>",
    trust_remote_code=True,
)
```

なお，この`revision`が固定するのはHugging Face上のデータセット読み込みスクリプトである。同スクリプトはWRIME原データをGitHubの`master`ブランチから取得するため，TSVファイル自体の完全な固定は別途対応が必要である。

根拠:

- [WRIMEデータセットリポジトリ](https://huggingface.co/datasets/shunk031/wrime/tree/main)
- [Datasets 2.21.0における`trust_remote_code`の処理](https://github.com/huggingface/datasets/blob/2.21.0/src/datasets/load.py#L109-L142)

### 3.2 論文どおりの総学習時間を測定できない

論文の`4_experimental_method.md:59`では，総学習時間を次のように定義している。

> 検証を含む3エポックの総学習時間を測定し，データ取得，前処理，チェックポイントの保存時間は含めない。

コードは`trainer.train()`が返す`train_result.metrics`を保存するため，Transformersが自動計算する`train_runtime`は記録される。しかし，Trainerは各エポックの検証とチェックポイント保存を実行した後に`train_runtime`を計算する。この値には検証時間だけでなく，論文では除外するとしたチェックポイント保存時間も含まれる。

そのため，現在の`train_runtime`をそのまま論文の総学習時間として報告すると，論文の定義と矛盾する。

対応方法は次のいずれかである。

- 論文の定義を「Trainerの学習処理全体の時間であり，検証とチェックポイント保存を含む」に変更する。
- CUDAを同期した独自の時間計測を実装し，チェックポイント保存時間を別途測定して除外する。

前処理や保存処理をどこまで含めるかによって結果が変わるため，LoRAとフルファインチューニングでは必ず同じ測定境界を使用する。

根拠:

- [Transformers 5.14.1の学習終了処理](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/trainer.py#L1828-L1843)
- [Transformers 5.14.1の評価・チェックポイント保存処理](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/trainer.py#L2061-L2103)

### 3.3 最大GPUメモリ使用量の計測

**対応済み（2026年7月31日）:** `trainer.train()`の直前にCUDA処理を同期してピーク統計をリセットし，学習終了直後に次の二つの値をGB単位で学習結果へ保存する処理を追加した。

- `max_gpu_memory_allocated_gb`: モデル，勾配，オプティマイザ状態，活性値など，PyTorchのテンソルが実際に使用した最大GPUメモリ
- `max_gpu_memory_reserved_gb`: PyTorchが再利用のために確保した領域を含む最大GPUメモリ

計測範囲は`trainer.train()`の呼び出し開始から終了までであり，学習，検証，チェックポイント保存，および最良チェックポイントの復元を含む。`trainer.train()`より後の最終モデル保存とテスト評価は含まない。主指標には`max_gpu_memory_allocated_gb`，参考値には`max_gpu_memory_reserved_gb`を使用する。LoRAにも同じ測定位置と処理を使用する必要がある。

根拠:

- [TransformersのTrainerメモリ計測に関する説明](https://huggingface.co/docs/transformers/v5.0.0/en/main_classes/trainer#transformers.TrainingArguments)

### 3.4 再現条件が十分に固定されていない

**一部対応済み（2026年7月31日）:** モデルとトークナイザに，同じモデルリポジトリの完全なコミットSHA`9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`を指定した。

```python
MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
)
model = Gemma3TextForSequenceClassification.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    ...
)
```

これにより，モデルリポジトリの`main`ブランチが更新された場合も，同じモデル重み，設定，およびトークナイザを取得できる。

`requirements.txt`の直接依存関係も，2026年7月31日時点で使用する次のバージョンへ固定した。

```text
accelerate==1.14.0
datasets==2.21.0
scikit-learn==1.9.0
torch==2.13.0
transformers==5.14.1
```

Python 3.12およびLinux環境を対象とした依存関係の解決に成功している。

**対応済み（2026年7月31日）:** 学習開始前に`outputs/full_finetune/experiment_environment.json`を生成し，次の情報を自動保存する処理を追加した。

- 実行日時
- Python，PyTorch，Transformers，Datasets，Accelerate，scikit-learnのバージョン
- CUDAとcuDNNのバージョン
- GPU名とGPUメモリ容量
- モデル，トークナイザ，データセットの名前とリビジョン

オプティマイザ設定の明示は未対応である。

次の対応が必要である。

- `optim="adamw_torch"`など，使用するオプティマイザをコードで明示する。
- Adamの係数，勾配クリッピング，スケジューラ，ウォームアップ，Weight Decayも論文に記録する。

根拠:

- [Transformers 5.14.1における既定オプティマイザの選択](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/training_args.py#L784-L793)

## 4. 論文とコードは一致するが，研究設計として改善が必要な点

### 4.1 乱数シードが1種類のみである

コードと論文はいずれも乱数シード42のみを使用している。分類ヘッドはランダムに初期化され，学習データの順序にも乱数が使われるため，1回の実験結果には偶然変動が含まれる。

LoRAとフルファインチューニングの性能差が小さい場合，その差が手法によるものか，初期値やデータ順序によるものかを判断できない。少なくとも3種類程度の乱数シードで実行し，QWK，Accuracy，MAE，学習時間について平均と標準偏差を報告することが望ましい。

また，`set_seed(42)`は乱数源を設定するが，GPU演算を含む完全な再現性を保証するものではない。PyTorchは，ライブラリのバージョンや実行環境をまたぐ完全な再現を保証していない。

根拠:

- [PyTorchの再現性に関する説明](https://docs.pytorch.org/docs/stable/notes/randomness)

### 4.2 ハイパーパラメータを各手法1設定だけ使用している

論文とコードでは，フルファインチューニングの学習率を`2e-5`，LoRAの学習率を`2e-4`としている。この違い自体は不適切ではないが，値を選択した根拠や検証手順が示されていない。

各手法に一つの設定だけを使用すると，観測された性能差に次の二つが混在する。

- ファインチューニング手法そのものの差
- 選択した学習率などが各手法にどの程度適していたかの差

検証データを用いた事前に定めた小規模な探索を行うか，先行研究や予備実験に基づく設定であることを明記する必要がある。探索を行う場合は，最終学習1回のコストとハイパーパラメータ探索全体のコストを分けて扱う。

### 4.3 FP16ではなくFP16混合精度学習である

`TrainingArguments(fp16=True)`は，モデルの全演算と全状態をFP16だけで保持する設定ではなく，自動混合精度による学習である。

したがって，論文の表にある「演算精度 FP16」は，次のように記述する方が正確である。

> FP16自動混合精度学習

根拠:

- [Accelerateの混合精度学習に関する説明](https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator)

### 4.4 最大入力長による切り捨ては発生しない

**確認済み（2026年7月31日）:** `check_truncation.py`を作成し，フルファインチューニングと同じ固定済みGemmaトークナイザでWRIME Ver.2の35,000件を実測した。512トークンを超える投稿は，学習，検証，テストのすべてで0件であった。

| 分割 | 総件数 | 512トークン超過 | 最大トークン長 |
|---|---:|---:|---:|
| 学習 | 30,000 | 0件（0.00%） | 192 |
| 検証 | 2,500 | 0件（0.00%） | 121 |
| テスト | 2,500 | 0件（0.00%） | 108 |
| 全体 | 35,000 | 0件（0.00%） | 192 |

したがって，`max_length=512`と`truncation=True`を指定しても，今回のWRIME Ver.2では投稿本文の情報は失われない。集計結果は`outputs/truncation_stats.json`へ保存した。

## 5. 軽微な論文との不一致

### 5.1 評価時に元のラベルへ戻していない

論文の`4_experimental_method.md:5`では，評価時にクラスIDを元の数値へ戻すと記載している。一方，`compute_metrics`は，正解と予測をクラスID 0から4のまま評価している。

ただし，変換は次のような一律の平行移動である。

```text
-2, -1, 0, +1, +2
 ↓   ↓  ↓   ↓   ↓
 0,  1, 2,  3,  4
```

Accuracyはラベルの一致だけを評価し，MAEの差は平行移動で変化しない。QWKの二次重みもクラス間距離に基づくため，この変換では値が変化しない。したがって，現在の実装でも評価結果は正しい。

実装を変更する必要はないが，論文を「クラスID 0から4の順序尺度上で評価する」と修正するか，コード側で明示的に元のラベルへ戻すと記述と処理が一致する。

## 6. 正しく実装されている点

### 6.1 データセットとラベル

- `load_dataset("shunk031/wrime", name="ver2")`によりWRIME Ver.2を指定している。
- 投稿本文として`sentence`を使用している。
- 正解ラベルとして`writer.sentiment`を使用している。
- ラベル−2，−1，0，+1，+2をクラスID 0，1，2，3，4へ順序を保って変換している。
- データセットが提供するtrain，validation，testをそのまま使用している。

公開されているWRIME Ver.2原データでも，分割数は学習30,000件，検証2,500件，テスト2,500件であり，各分割間の書き手の重複はなかった。

根拠:

- [WRIMEデータセットカード](https://huggingface.co/datasets/shunk031/wrime)
- [WRIME Ver.2データセットスクリプト](https://huggingface.co/datasets/shunk031/wrime/blob/main/wrime.py)

### 6.2 モデルと学習対象

- 5クラスの文章分類ヘッドを追加している。
- `problem_type="single_label_classification"`を指定している。
- 全パラメータ数と学習可能パラメータ数が一致することを学習前に検証している。
- 学習に不要なKVキャッシュを無効化している。

`Gemma3TextForSequenceClassification`は，Gemma 3 270Mのようなテキスト専用の小規模Gemma 3モデルに対応する正式な文章分類クラスであり，選択は適切である。

根拠:

- [TransformersのGemma 3モデル文書](https://huggingface.co/docs/transformers/v5.13.1/en/model_doc/gemma3)

### 6.3 評価指標

- QWKを`weights="quadratic"`で計算している。
- ラベル順を0から4に固定している。
- Accuracyを計算している。
- MAEを順序付きクラスID上で計算している。
- QWKを最良モデル選択の主指標としている。

`metric_for_best_model="qwk"`は，Trainerが付与する`eval_`接頭辞を省略した有効な指定である。

### 6.4 データ利用とモデル選択

- 学習データだけでパラメータを更新している。
- 各エポック終了時に検証データを評価している。
- 検証QWKが最大のチェックポイントを復元している。
- テストデータは最良モデル決定後に一度だけ評価している。
- テストデータをモデル選択に使用していない。

### 6.5 その他

- 乱数シードをモデル初期化前に設定している。
- 学習用シードとデータ用シードを明示している。
- ミニバッチ内の最長系列に合わせた動的パディングを使用している。
- 学習率，エポック数，バッチサイズ，最大入力長は論文の表と一致している。
- 検証とチェックポイント保存の間隔は，論文どおり各エポックである。

## 7. TODO

### 7.1 コードを修正するTODO

上から順に対応する。論文側で測定方法を定義する必要がある項目は，対応する論文TODOを確定してから実装する。

- [x] `load_dataset`へHugging Faceデータセットリポジトリの`revision`と`trust_remote_code=True`を追加する。
- [ ] 学習時間の計測処理を実装する。主指標は学習と検証に要した時間とし，チェックポイント保存時間を除外する。Trainerが出力する`train_runtime`と，CUDA同期を行って別途計測した`checkpoint_save_runtime`を保存し，`training_runtime_excluding_save = train_runtime - checkpoint_save_runtime`として算出する。確認用の参考値として，`train_runtime`と`checkpoint_save_runtime`も結果へ保存する。LoRAとフルファインチューニングには同じ計測処理を使用する。
- [x] 学習時の最大GPUメモリを計測する。測定開始前にピーク統計をリセットし，`max_gpu_memory_allocated_gb`と`max_gpu_memory_reserved_gb`を結果へ保存する。
- [x] モデルとトークナイザの`revision`をコミットSHA`9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`で固定する。
- [x] `requirements.txt`のPyTorch，Accelerate，scikit-learnを最終実験で使用するバージョンへ固定する。
- [x] Python，PyTorch，Transformers，Datasets，Accelerate，scikit-learn，CUDA，cuDNN，GPU名，GPUメモリ容量，モデル・トークナイザ・データセットのリビジョン，実行日時を`outputs/full_finetune/experiment_environment.json`へ自動保存する。
- [x] `check_truncation.py`を実行し，最大入力長512トークンを超える投稿が全分割で0件であることを確認して，`outputs/truncation_stats.json`へ保存する。

### 7.2 論文を修正するTODO

- [ ] 総学習時間の測定範囲を確定する。検証時間を含めるか，チェックポイント保存時間を除外するか，モデルやオプティマイザの初期化時間を含めるかを明記する。
- [ ] 最大GPUメモリ使用量について，主指標に`max_gpu_memory_allocated_gb`，参考値に`max_gpu_memory_reserved_gb`を用いることを記載する。計測範囲は`trainer.train()`の呼び出し開始から終了までとし，学習，検証，チェックポイント保存，および最良チェックポイントの復元を含むことも明記する。
- [ ] 最終実験で固定したソフトウェアのバージョン，CUDA環境，GPU，モデル・トークナイザ・データセットのリビジョン，実行日時を実験環境へ記載する。
- [ ] オプティマイザ，Adamの係数，勾配クリッピング，学習率スケジューラ，ウォームアップ，Weight Decayを学習条件へ追記する。
- [ ] 「演算精度 FP16」を「FP16自動混合精度学習」へ修正する。
- [ ] 「評価時に元の数値へ戻す」という記述を，実装に合わせて「クラスID 0から4の順序尺度上で評価する」へ修正する。コード側で元のラベルへ戻す場合は，この修正は不要である。
- [ ] 複数の乱数シードで実験し，QWK，Accuracy，MAE，学習時間について平均と標準偏差を報告する実験計画へ変更する。
- [ ] LoRAとフルファインチューニングのハイパーパラメータを選択した根拠または検証データを用いた探索方法を記載する。探索を行う場合は，最終学習のコストと探索全体のコストを区別する。
