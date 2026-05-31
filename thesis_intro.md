## タイトル 
WRIME Ver.2 を用いた LLM による SNS 投稿の感情分析
— ベースモデル・LoRA・フルファインチューニングの比較 —

# 1. 序論

## 1.1 研究背景

大規模言語モデル（LLM）は，翻訳・要約・質問応答など多様な自然言語処理タスクで高い性能を示している（Brown et al., 2020）。また，感情分析においてもその有効性が報告されており，従来手法を上回る性能を示す例が増えている（Zhang et al., 2024）。

LLM を特定のタスクへ適応させるため，モデル全体のパラメータを更新するフルファインチューニングが広く用いられている。しかし，フルファインチューニングでは全パラメータを学習対象とするため，多大な計算資源が必要となる。この課題を解決する手法として，少数の追加パラメータのみを学習する LoRA（Low-Rank Adaptation）が提案されている（Hu et al., 2022）。

日本語においても LoRA を用いた研究が進みつつある。王・中町・佐藤（2023）は，日本語 GPT モデル（6.7B）に対してフルファインチューニングと LoRA を比較し，フルファインチューニングが 6,666M の学習パラメータと約 483GB の GPU メモリを必要としたのに対し，LoRA は 16M の追加パラメータと 54MB のメモリで済みながら，含意認識タスクにおいて同等以上の性能を示したことを報告している。また，助田・鈴木・坂地・小寺（2024）は，医療ドメインの日本語 LLM に LoRA を適用し，専門知識を効率的に組み込める可能性を示している。

しかし，これらの研究の多くは感情分析を対象としておらず，日本語感情分析におけるベースモデル，LoRA，およびフルファインチューニングの比較検証は十分に行われていない。

## 1.2 研究目的

本研究の目的は，SNS 投稿を対象とした感情分析において，学習を行わないベースモデル，LoRA，およびフルファインチューニングの三手法を比較し，性能および計算コストの観点から各手法の特徴を明らかにすることである。

## 1.3 研究方法

本研究では，日本語 SNS 投稿に対して感情極性ラベルが付与されたデータセットである WRIME Ver.2（Suzuki et al., 2022）を用いる。WRIME Ver.2 は，書き手本人による主観的な感情極性ラベルと，読者による客観的な感情極性ラベルが付与されたデータセットであり，WRIME（Kajiwara et al., 2021）を拡張したものである。本研究では，このうち書き手本人による感情極性ラベルである Writer_Sentiment を使用する。データセットは WRIME Ver.2 の公式分割に従い，学習用 30,000 件，検証用 2,500 件，テスト用 2,500 件を使用する。なお，各分割間で書き手は重複しない。

Writer_Sentiment として付与されている感情極性ラベル（−2〜+2）を用いて感情極性分類を行うタスクを設定する。具体的には，「−2（強いネガティブ）」「−1（ややネガティブ）」「0（中立）」「+1（ややポジティブ）」「+2（強いポジティブ）」の 5 段階からなる感情極性を予測対象とする。本研究では，これらのラベルを予測する 5 クラス分類タスクとして扱う。感情極性分類を行うため，モデルには 5 クラス分類用の分類ヘッドを付加し，各ラベルを予測する。

使用するモデルには，Google によって公開された大規模言語モデルである Gemma 3（270M）を採用する。このモデルは，2 億 7,000 万パラメータという小規模ながら，事前学習によって指示追従能力と構造化テキスト処理能力が付与されているほか，タスク固有のファインチューニング用途を前提に設計されている点に特徴がある（Lacombe et al., 2025）。また，Gemma 3 は計算資源の限られた環境でも動作可能な設計がなされており，性能と計算コストの両面から比較を行う本研究の目的に適していると判断した。

比較対象とする手法は，学習を行わないベースモデル，LoRA によるパラメータ効率の高いファインチューニング，およびフルファインチューニングの三種類とする。各手法において感情極性分類を行い，性能および計算コストを比較する。

評価指標には Quadratic Weighted Kappa（QWK）を主指標として用いる。感情極性ラベルには順序関係が存在するため，QWK は予測ラベルと正解ラベルの距離を考慮した評価が可能であり，本タスクの評価に適している。また，補助指標として Accuracy および Mean Absolute Error（MAE）も算出する。さらに，各手法における学習時間および学習可能パラメータ数を比較し，性能と計算コストのトレードオフについて評価する。

## 文献リスト

- Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... Amodei, D. (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*.
- 王 昊・中町 礼文・佐藤 敏紀 (2023). 日本語の大規模な基盤モデルに対する LoRA チューニング 言語処理学会第 29 回年次大会発表論文集, 2222-2226.
- Lacombe, O., Kenealy, K., Black, K., Kumar, R., Visin, F., & Zhang, J. “Gemma 3 270M の概要: 超高効率 AI のためのコンパクトモデル”. Google for Developers. 2025 年 8 月 14 日. https://developers.googleblog.com/ja/introducing-gemma-3-270m/, (2025 年 10 月 20 日)
- 助田 一晟・鈴木 雅弘・坂地 泰紀・小寺 聡 (2024). JMedLoRA：Instruction-tuning による日本語大規模モデルの医療ドメイン適用 言語処理学会第 30 回年次大会発表論文集, 2548-2553.
- Zhang, W., Deng, Y., Liu, B., Pan, S. J., & Bing, L. (2024). Sentiment Analysis in the Era of Large Language Models：A Reality Check. *Findings of the Association for Computational Linguistics: NAACL 2024*, 3881-3906.
- 山田育矢 (2023). 大規模言語モデル入門, 技術評論社.
- Kajiwara, T., Chu, C., Takemura, N., Nakashima, Y., & Nagahara, H. (2021). WRIME: A New Dataset for Emotional Intensity Estimation with Subjective and Objective Annotations. *Proceedings of NAACL-HLT 2021*, 2095–2104.
- Suzuki, H., Miyauchi, Y., Akiyama, K., Kajiwara, T., Ninomiya, T., Takemura, N., Nakashima, Y., & Nagahara, H. (2022). A Japanese Dataset for Subjective and Objective Sentiment Polarity Classification in Micro Blog Domain. *Proceedings of LREC 2022*, 7022–7028.
