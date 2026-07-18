# 文献メモ

## 書誌情報

- 著者: Haruya Suzuki, Yuto Miyauchi, Kazuki Akiyama, Tomoyuki Kajiwara, Takashi Ninomiya, Noriko Takemura, Yuta Nakashima, Hajime Nagahara
- 年: 2022
- タイトル: A Japanese Dataset for Subjective and Objective Sentiment Polarity Classification in Micro Blog Domain
- 掲載先: Proceedings of LREC 2022, 7022-7028
- URL: https://aclanthology.org/2022.lrec-1.759/
- PDF: suzuki_2022_wrime_ver2.pdf

## 概要

WRIMEを拡張し，日本語SNS投稿35,000件に対して，基本感情の強度ラベルと感情極性ラベルを，書き手本人の主観的観点と読み手の客観的観点の両方から付与したデータセットを提案した論文である。感情極性ラベルは，strong negative, negative, neutral, positive, strong positive の5段階であり，本研究で用いる `-2, -1, 0, +1, +2` のラベル設定に対応する。

本論文では，感情強度と感情極性の関係も分析している。Plutchikの8感情のうち，喜び，期待，信頼はポジティブ方向，悲しみ，怒り，恐れ，嫌悪はネガティブ方向と相関し，驚きは明確にどちらか一方には属さないとされる。また，読者は書き手の感情極性をおおまかには推定できるが，強いポジティブ・強いネガティブのような極端な強度を弱めに推定する傾向がある。

実験では，35,000件を学習30,000件，検証2,500件，テスト2,500件に分割し，書き手が分割間で重複しないようにしている。評価指標としてAccuracy，MAE，QWKを用い，BERTベースのモデルがBoW+LogRegより高い性能を示した。また，書き手本人の主観的感情極性の推定は，読者による客観的感情極性の推定より難しいことが示されている。

## 本研究との関係

本研究で使用する中心的なデータセット文献である。2.2「WRIMEおよびWRIME Ver.2」と3.2「使用データセット」で必ず引用する。本研究では，このデータセットのうち書き手本人による主観的感情極性ラベル `writer.sentiment` を用い，5クラス感情極性分類として扱う。

## 使えそうな記述

- WRIME Ver.2は，日本語SNS投稿35,000件に，基本感情と感情極性のラベルを付与したデータセットである。
- 感情極性は，強いネガティブ，ネガティブ，中立，ポジティブ，強いポジティブの5段階である。
- 書き手本人による主観的ラベルと，読み手による客観的ラベルの両方が付与されている。
- 公式実験では，学習30,000件，検証2,500件，テスト2,500件に分割し，書き手の重複を避けている。
- 評価指標としてAccuracy，MAE，QWKが用いられている。

## 注意点

WRIME Ver.2には主観・客観の両方のラベルがあるが，本研究では `writer.sentiment` のみを使う。本文では，読者側の客観ラベルと混同しないように，「書き手本人による主観的感情極性ラベル」と明記する。
