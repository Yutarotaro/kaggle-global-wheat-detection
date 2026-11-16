# 実験記録

## 2026-11-14: Kaggle公式データのGPUベースライン

KaggleのTesla T4でMobileNetV3 Faster R-CNNを画像サイズ640、3 epoch、バッチ4、シード42で学習した。
撮影元別の固定分割は学習2,002枚、検証636枚、最終確認用735枚。注釈・撮影元がない49枚は除外し、
最終確認用データは未評価。

13:28:49（日本時間）のローカル検証スコアは **0.445099**。撮影元別では
`rres_1` が0.558528、`arvalis_2` が0.204897。
[Kaggle実行 v1](https://www.kaggle.com/code/yutarotaro/ml-loop-10a1829d2fc84c67-0001) は完了し、
実行 `10a1829d2fc84c67` に重み、予測、評価結果を回収した。この結果をベースラインとして、
4時間の改善実験 `f5d6ac6937e943b8` を開始した。

同じベースラインの重みを使い、ネット接続なしの[非公開推論Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-baseline-late-20261114)を用意した。
提出用CSVをKaggle上で生成できることを確認し、終了後の採点へ1回だけ提出した（提出ID `58331537`）。
採点結果はPublic **0.5358**、Private **0.4449**。提出スコアはモデル選択や改善ループの入力には使わない。
提出時のコードは `notebooks/submission-baseline.ipynb` に保存した。
各実験のコードは実行ディレクトリ内の `artifacts/<trial-id>/bundle/experiment.ipynb` を参照する。
旧方式の実験は、送信済みのバンドルからNotebookを復元した。元の実行やスコアとは区別して参照する。

## 2026-11-14: 12 epochのGPU学習

`8fc98f3e68e8441a-0001` は640px、バッチ4、横反転確率0.50、シード42で12 epoch学習し、
固定検証で **0.512961**。valid lossの最小値はepoch 4の0.96384で、epoch 12は1.01663。
各epochの検出スコアは採点していない。次の比較では6 epochを基準にする。
保存済み検証予測のconfidenceを0.30から0.50へ上げると探索用検証スコアは **0.525791**。
同じ検証分割での調整結果で、holdoutは未評価。

## 試行の系譜と次の比較

学習の派生とアンサンブルへの合流は [実験系譜](experiment-lineage.md) を参照。

| 試行 | 比較元 | 試したアイデア | 固定検証 | 親との差 |
| --- | --- | --- | ---: | ---: |
| `10a1829d2fc84c67-0001` | 基準 | MobileNetV3 Faster R-CNN、640px、3 epoch、横反転 | 0.445099 | — |
| `f5d6ac6937e943b8-0002` | `10a1829d2fc84c67-0001` | 背景の分類損失を1.25倍にして誤検出を減らす | 0.451619 | +0.006519 |
| `f5d6ac6937e943b8-0003` | `f5d6ac6937e943b8-0002` | 枠の回帰損失を1.20倍にして位置ずれを減らす | 0.450926 | −0.000693 |
| `f5d6ac6937e943b8-0004` | `f5d6ac6937e943b8-0002` | 正例の分類損失を1.10倍にして見逃しを減らす | 0.439911 | −0.011708 |
| `f5d6ac6937e943b8-0005` | `f5d6ac6937e943b8-0002` | 背景の分類損失の重みを序盤20%で1.00倍から1.25倍へ上げる | 0.448220 | −0.003399 |
| `8fc98f3e68e8441a-0001` | `10a1829d2fc84c67-0001`（別セッションの参考） | 3から12 epochへ延ばし、過学習の兆候も観測する | 0.512961 | +0.067861 |
| `8fc98f3e68e8441a-0002` | `8fc98f3e68e8441a-0001` | 色・明るさ補正0.15で撮影元の違いに強くなるか検証する | 0.509165 | −0.003796 |

背景損失の固定係数には小幅な改善があった。追加の損失調整3案はその親より悪かったため、
次は学習エポック数、augmentation、解像度の比較を優先する。6 epochの横反転のみを基準に、
色補正と上下反転をそれぞれ追加する。色補正では全体のスコアに加え撮影元別の値を確認する。
confidence 0.50の結果は同じ予測の再採点で、追加学習の効果とは区別する。

実行時の全試行と最新スコアは、MLflow同期が `.loop-wheat/experiments.md` に自動記録する。
ここには比較から得られた判断と次に試す条件を追記する。

## 2026-11-14: 検出器とデータの見直し

12 epochの基準試行 `8fc98f3e68e8441a-0001` は、固定検証で全体0.512961、
撮影元別では `rres_1` が0.607486、`arvalis_2` が0.312789。IoU 0.50での再現率は
小さい穂が10.1%、中サイズが85.0%、大サイズが84.9%。各画像の検出数は最大77件で、
設定上の上限300件に達した画像はなかった。

固定検証から撮影元ごとに32枚をシード42で抽出し、同じ640px学習済み重みをMacのCPUで
640pxと1024pxで推論した。confidence 0.30、NMS 0.50。640pxで回収済みのKaggle予測と同じスコアを再現した。

| 推論解像度 | 64枚のスコア | 小さい穂の検出数（IoU 0.50） | rres_1 | arvalis_2 |
| --- | ---: | ---: | ---: | ---: |
| 640px | 0.444753 | 7 / 78 | 0.594699 | 0.294807 |
| 1024px | 0.295867 | 24 / 78 | 0.467532 | 0.124201 |

解像度変更のみで小さい穂の検出は増えたが、総合スコアは両撮影元で低下した。
次の候補は、強い単体検出器を768〜1024pxで学習から揃えた基準と、
同じ検出器にMosaicを加えた比較。固定検証の全体、撮影元別、小さい穂の再現率を合わせて判断する。

DungNBの1位解法は、外部データを組み込んだ学習、MosaicとMixUp、EfficientDetとFaster R-CNNの
複数解像度での学習、TTAとWBF、テスト画像の擬似ラベルを組み合わせた。
擬似ラベルを試す場合は対象画像への適応と記録し、未見データへの汎化を測る固定検証と区別する。
[1位の公開コード](https://github.com/dungnb1333/Global-Wheat-Dection-1st-Place) ·
[主催者による分析](https://pmc.ncbi.nlm.nih.gov/articles/PMC10795497/)

次の比較は `loop.efficientdet.toml` で実行する。COCO重みからEfficientDet-D2を768pxで
6 epoch学習し、12 epochの640px MobileNetV3を参照値とする。この比較には検出器、解像度、
エポック数の差がある。続く試行はD2の基準Runを親にし、Mosaicの確率だけを0から0.50へ変更する。
固定検証の全体、撮影元別、小さい穂の再現率を記録する。

### 学習・検証データの分布

[データ分布の集計](data-profile.md)に、固定分割の撮影元別の集計表と再現手順を記録した。

穂数の中央値は学習50本、検証37本。25本未満の画像が学習では101枚（5.0%）、
検証では173枚（27.2%）で、`arvalis_2` に156枚（撮影元内の76.5%）が集中する。
`arvalis_2` で枠の面積が画像の0.2%未満の穂は49 / 4,179本（1.2%）。

EfficientDet-D2基準のIoU 0.50での誤検出は、`arvalis_2` で2,240件（11.0件/枚）、
`rres_1` で2,343件（5.4件/枚）。穂数25本未満にそろえても、スコアはそれぞれ
0.285（156枚）、0.621（17枚）だった。撮影元間の差を穂数だけで説明できるかは、
サンプル数の異なるこの比較だけでは判断できない。

次のデータ実験では、学習内で穂数25本未満の画像を多めにサンプリングし、
同じ学習ステップ数の通常サンプリングと比較する。固定検証で撮影元別・穂数別の
誤検出と見逃しを記録する。

## 2026-11-14 保存済み予測のアンサンブル

640px・12 epochの横反転のみ（`8fc98f3e68e8441a-0001`）と色補正0.15を追加した試行
（`8fc98f3e68e8441a-0002`）の予測を統合した。両モデルの枠が重なるときに座標を加重平均し、
位置ずれ・重複検出が減るかを、撮影元別の固定検証636枚で確認した。
Weighted Boxes Fusion（WBF）の枠統合IoUは0.55。入力は各モデルの保存済み予測（confidence 0.30以上）、
出力confidenceは0.50。 [WBF論文](https://arxiv.org/abs/1910.13302)

| 条件 | 固定検証 | rres_1 | arvalis_2 | 小さい穂の再現率（IoU 0.50） |
| --- | ---: | ---: | ---: | ---: |
| 横反転のみ、confidence 0.50 | 0.525791 | 0.609015 | 0.349552 | 3.87% |
| 色補正あり、confidence 0.50 | 0.515252 | — | — | — |
| WBF、均等 | 0.532910 | 0.614321 | 0.360511 | 2.82% |
| WBF、横反転のみを2倍 | 0.532010 | 0.614012 | 0.358361 | 3.13% |

均等重みでは同じconfidenceの最良単体より +0.007119。IoU 0.50で誤検出は2,385件から2,147件に、
重複は93件から31件に減った。見逃しは5,523件から5,708件に増えた。
均等重みのconfidence探索では0.45が最高の0.532990だった。最終確認用分割は未評価。
横反転のみを2倍にすると小さい穂の再現率がやや戻り、全体は均等重みを0.000900下回った。
次の比較はEfficientDet-D2単体と、横反転のみのFaster R-CNNとの均等重みWBF。
同じ検証画像で各モデル単体のconfidenceを比較し、小さい穂の見逃しも確認する。

再現コードはコミット `29b9e6d`。親の色補正試行のソースは凍結済みスナップショット
（`source_digest` は `00e89791979289f86eb74eeefc1a591121bea88a078240c6c6aad4c81e698fef`）を参照する。
設定、固定分割と入力予測のハッシュ、confidence別の結果は
`.loop-wheat/ensemble/mbv3-hflip-color15/comparison.json` と
`.loop-wheat/ensemble/mbv3-hflip-color15-2to1/comparison.json` に保存した。
ローカルMLflowで両方の学習Run、固定検証4枚の検出画像、エラー例を参照できる。
[均等重みのMLflow Run](http://localhost:5050/#/experiments/1/runs/09a62b14beb44bc89a64632302d6dd19) ·
[2:1のMLflow Run](http://localhost:5050/#/experiments/1/runs/1c2a456edb9f4e5eaa6a24ecd171cfa0)

## 2026-11-14 Faster R-CNNとEfficientDet-D2のアンサンブル

`8fc98f3e68e8441a-0001` と `d31623fdd5114e49-0001` を同じ固定検証636枚で比較。
均等重み、統合IoU 0.55のWBFは、出力confidence 0.50で **0.569306**。
同じ検証で選んだconfidence 0.40の再採点は **0.580485**。
同じ0.40でD2単体は0.544813（差 +0.035672）、Faster R-CNN単体は0.522629。
WBFの撮影元別スコアは `rres_1` 0.675937、`arvalis_2` 0.378352。
小さい穂の再現率はIoU 0.50で5.96%。ローカルの最終確認用分割は未評価。
[MLflowのアンサンブルRun](http://localhost:5050/#/experiments/1/runs/3600138dbd634b52b2e050d204d9ee51)

同じ条件の推論専用Notebook v1をネット接続なしで実行し、Kaggleのサンプル10枚でCSVを確認。
終了後の採点に提出した（提出ID `58342094`、提出ソース `98308fd`）。
採点結果は **Public 0.6512 / Private 0.5466**。
初回提出 `58331537` からの差は Public +0.1154、Private +0.1017。
学習元は `8fc98f3e68e8441a-0001` と `d31623fdd5114e49-0001`。
[Kaggle Notebook](https://www.kaggle.com/code/yutarotaro/wheat-infer-d2-mbv3-wbf040) ·
[MLflowの提出推論Run](http://localhost:5050/#/experiments/1/runs/6bc49f5820c848789b24245c54089ab0)

## 2026-11-14 Arvalis Monsの追加データ試行

D2基準の `d31623fdd5114e49-0001` と比較する。同じ初期重み、768px、6 epoch、
バッチ4、seed 42、各epoch 2,002画像・501更新ステップ。
追加試行では既存2,002枚とGWHD 2021のモン60枚から、各epochに2,002枚を重複なしで抽出する。
固定検証の全体スコアと `arvalis_2` の誤検出件数で比較する。
[取得・選定記録](data-profile.md#追加データの取得と選定) ·
[Kaggleの追加データ](https://www.kaggle.com/datasets/yutarotaro/gwhd2021-mons-wheat)

初回の実行 `6e58debdff994e7d` は、文書の図をNotebookソースとして読み込もうとしたため、
Kaggle投入前に停止した。文書の図を同梱対象から外す設定を追加した上で、別の実行IDでやり直す。

```bash
../ml-research-loop/.venv/bin/ml-loop run --project . --config loop.efficientdet-mons.toml \
  --state .loop-wheat --budget 2h --trials 1
```

## 2026-11-14: GWHD 2020 v4のローカル接続確認

[公式公開版（Zenodo 4298502）](https://zenodo.org/records/4298502)の注釈付き画像から、
7撮影元を各2枚、シード42で選択した。学習8枚、検証4枚、holdout 2枚に撮影元で分割し、
holdoutは未評価。MacのCPUでMobileNetV3 Faster R-CNNを画像サイズ256、1 epoch、バッチ2で学習した。

検証スコアは **0.004615**。Kaggle取得は403で、これは公開修正版の14枚だけを使った接続確認。
Kaggleのスコアや全データのベースラインには使用しない。設定、分割、重み、評価値、
誤検出画像はローカルの `data/gwhd2020-v4/smoke-project/` に保存した。

## 2026-11-15 YOLOX-Sの単体とWBF

`602ac2d18a3042dc-0001` はCOCOで事前学習したYOLOX-Sを768pxで6 epoch学習した。
元のKaggle画像2,002枚を毎epoch入力し、確率0.5で横反転。固定検証ではconfidence 0.30で
**0.345790**、confidence 0.40で **0.345630**。小さい穂の再現率は0.30で19.44%。
confidence 0.30の撮影元別スコアは `rres_1` が0.439855、`arvalis_2` が0.146592。
同じconfidence 0.40と均等重みのWBFで、D2＋YOLOX-Sは **0.477607**、
D2＋Faster R-CNN＋YOLOX-Sは **0.557939**。既存のD2＋Faster R-CNN **0.580485** を次の基準に使う。

YOLOX-S単体をconfidence 0.30でKaggleに提出した（ID `58362730`、推論Notebook v1、
commit `dcb26bd`）。採点は **Public 0.5645 / Private 0.5005**。
[実験系譜](experiment-lineage.md#kaggleへの提出) ·
[学習Run](http://localhost:5050/#/experiments/1/runs/4187c38c343e49c3a7dfdd9fd66dd85f)

## 候補の採否と次の仮説

### 上位解法を踏まえた比較順

2020年1位のDungNBは外部データでの学習、複数スケール、EfficientDetとFaster R-CNNのWBF、
疑似ラベルを併用した。2位OverFeatはCOCO事前学習済みEfficientDet-D6を主軸とし、枠のノイズ、
画像端で切れた穂、色・幾何変換、TTAを調べた。9位の実装は二つの検出器のconfidenceの分布差を
補正してからWBFを行った。主催者によるEfficientDet-D4の比較では、単画像のaugmentationで
撮影元均等の指標が0.4528から0.4981、MixUp単独では0.3675だった。
[1位の実装](https://github.com/dungnb1333/Global-Wheat-Dection-1st-Place) ·
[2位の解説](https://www.kaggle.com/competitions/global-wheat-detection/writeups/overfeat-2nd-place-solution-with-code-mit-complian) ·
[9位の実装](https://github.com/amirassov/kaggle-global-wheat-detection) ·
[主催者の比較実験](https://arxiv.org/pdf/2105.06182#page=11)

1. D2の同じ初期重み、768px、更新回数、seedで、元データ2,002枚と
   [追加候補385枚を加えたデータ](data-profile.md#gwhd-2021の追加学習候補を監査)を比較する。
   前回はKaggle上で追加ZIPを見つけられなかったため、アップロード後のファイル名と枠の読み込みを
   本学習前に確認する。各epochで同じ固定検証を採点し、
   最高スコア時の重みと最終重みを保存する。
2. データの比較を終えてから、上下反転・回転・枠と連動するcrop・明暗・ぼけを含む
   単画像augmentationを一条件として追加する。D2の現行Mosaic 0.50は基準0.526891から
   0.512726に低下したため、複数画像を混ぜる方法は個別の実験として扱う。
3. 採用した学習条件で、モデルサイズと解像度を順に比較する。既存重みを使う左右・上下反転TTAと
   モデル別confidenceの調整は、学習条件の比較とは別に評価する。
   小さい穂の見逃しが残る場合は重なり付きタイル推論を調べ、境界の重複と誤検出も数える。

比較では全体、撮影元別、小さい穂の再現率、誤検出数を追う。最終確認とKaggleへの提出には、
固定検証で改善し、撮影元別の条件も満たした候補を使う。

### 採用基準

同じセッションで採用済みの最良Runを基準に、固定検証の全体スコアが上がり、
どの撮影元でも悪化が0.01以内なら採用する。撮影元別の悪化が0.01を超えた候補は保留とする。
小さい穂の再現率と見逃し・誤検出・重複の件数は次の仮説の判断材料に使う。

採否、理由、基準Runと次に調べる点はLoopの状態に保存し、MLflowのDescriptionと
`.loop-wheat/experiments.md` に同期する。次の仮説提案にはこれらの記録と検証の集計を渡す。
GPUやデータの問題でスコアが出なかった試行も、失敗の詳細を実験履歴に残す。

### 385枚追加の学習比較を再現

元画像の2,002枚と、追加385枚を含む2,387枚を比較する。追加群では各epochに2,002枚を
重複なしで抽出し、両群の更新回数を3,006ステップにそろえる。固定検証を各epochで採点し、
`weights.pt` に最良の重み、`weights-last.pt` に最終重み、
`checkpoint-selection.json` に最良epochを保存する。比較はどちらも6 epochまで行う。

```bash
uv run --locked python scripts/prepare_gwhd2021_training.py \
  data/gwhd2021/gwhd2021jpg.zip --output data/gwhd2021/field385
../ml-research-loop/.venv/bin/ml-loop run --project . --config loop.efficientdet-field385.toml \
  --state .loop-wheat --budget 8h --trials 2
```

追加アーカイブのSHA-256は `ce75e5496e61ec25254423fdd2dc060666b7d67139ca5401e8102137b2598997`。
Kaggleにマウントした保管ファイルを同じSHA-256で確認する。
[追加データ](https://www.kaggle.com/datasets/yutarotaro/wheat-gwhd2021-field385) ·
[マウントの確認Notebook](https://www.kaggle.com/code/yutarotaro/wheat-field385-mount-check)

#### 385枚追加の結果

追加候補をKaggleで照合し、385枚・21,720枠を読み込めることを確認した。
条件はD2、768px、seed 42、各epoch 2,002画像、6 epoch、最良epoch選択。

| 固定検証636枚 | 基準 `1d692cda0f6e4cd2-0001` | 追加 `1d692cda0f6e4cd2-0002` |
| --- | ---: | ---: |
| スコア | 0.539742 | 0.547748 |
| 最良epoch | 3 | 6 |
| `rres_1` | 0.656441 | 0.668488 |
| `arvalis_2` | 0.292616 | 0.292062 |
| 小さい穂の再現率（IoU 0.50） | 23.20% | 30.51% |
| 誤検出（重複を除く、IoU 0.50） | 5,046 | 4,376 |

追加D2と既存Faster R-CNNを均等WBF（IoU 0.55、confidence 0.40）で統合すると
0.586916。D2を2倍にしたWBF（IoU 0.45、confidence 0.40）は0.597387だったが、
`arvalis_2` は0.366327で、既存最良アンサンブルの0.378352からの低下が0.01を超えた。
均等WBFの `arvalis_2` は0.375622。
[均等WBF](http://localhost:5050/#/experiments/1/runs/25e55cf5f0214550b5c0866b220e7f9d) ·
[D2を2倍にしたWBF](http://localhost:5050/#/experiments/1/runs/360a0d4b1c8f4e1ebb25227ef40490a0)

#### 既存重みのWBFを調整

既存D2とFaster R-CNNの保存済み予測を使い、IoU 0.45 / 0.55 / 0.65、
Faster R-CNN対D2の重み1:1 / 1:2 / 2:1、および出力confidenceを比較した。
現在の採用基準を満たすうち最高の条件は **IoU 0.45、重み1:2、confidence 0.40**。
固定検証は0.580485から **0.587241**、候補確定後に別途評価した最終確認735枚では
0.555930から **0.557030**。最終確認の `arvalis_3` と `inrae_1` は両方改善した。
固定した推論Notebook v1（commit `64ba330`）を提出した。提出IDは `58370638`。
Kaggleの確定値は **Public 0.6668 / Private 0.5651**。
提出 `58342094` からPublicは+0.0156、Privateは+0.0185。
[比較Run](http://localhost:5050/#/experiments/1/runs/3ddf9f9ca828466dba74651f97c1b183) ·
[提出Run](http://localhost:5050/#/experiments/1/runs/de5bd96bb3964629938b13ec172dd35a) ·
[提出Notebook](https://www.kaggle.com/code/yutarotaro/wheat-infer-d2-mbv3-wbf045-d2x2)

## 2026-11-15 推論構成の比較

比較元は提出 `58370638`（D2＋MobileNetV3 Faster R-CNN、重み1:2、WBF IoU 0.45）。
固定検証636枚で、IoU 0.50で見逃した4,744穂のうち2,082穂には統合前の
候補にIoU > 0.50の枠があった。面積が画像の0.2%未満の穂は893穂を見逃し、
そのうち682穂では統合前でも最大IoUが0.30未満だった。候補の選択と、解像度を
上げて候補を増やす方法を比較した。学習済み検出器は毎回同じ版を使用した。

| 推論方法 | 固定検証 | arvalis_2 | rres_1 | 独立確認 |
| --- | ---: | ---: | ---: | ---: |
| 提出 `58370638` | 0.587241 | 0.368985 | 0.690307 | 0.557030 |
| 両モデルの水平反転TTA | **0.598597** | 0.378439 | 0.702560 | **0.564296** |
| D2の4タイル候補を追加 | 0.572241 | 0.343029 | 0.680480 | — |
| 全体画像の候補を学習した枠品質で再選択 | 0.544161 | 0.316172 | 0.651822 | — |
| 学習した枠品質で既存枠を補正しタイル候補を追加 | 0.558535 | 0.326579 | 0.668070 | — |
| 全体画像とタイルの全候補を枠品質で再選択 | 0.497794 | 0.257072 | 0.611468 | — |

水平反転TTAは、各モデルの元画像と反転画像の候補を元の座標に揃え、WBF
IoU 0.55で統合した。その後、Faster R-CNNとD2を1:2で統合し、提出時の下限は
0.34とした。独立確認735枚の内訳は `arvalis_3` 0.529517→0.536762、
`inrae_1` 0.644416→0.651745。提出Notebook v1（commit `53b276a`、
提出ID `58374318`）は **Public 0.6772 / Private 0.5719**。

タイルでは、画像辺の65%の窓を四隅から採り、クロップの内側の境界から画像辺の
2%以内にかかる枠を除外した。D2の入力は768px。候補追加の試行では、
全体画像で採用した枠とのIoUが0.30未満でconfidenceが0.50以上の小さい枠を候補とした。
固定検証用の候補はKaggle Notebook v1（commit `809b59a`）で取得した。
小さい穂の再現率は6.7%から15.2%へ増え、見逃しは4,744件から4,151件に減った。
誤検出が1,765件から2,946件に増え、固定検証総合点は低下した。

枠品質は、両方の検出器が未使用のGWHD 2021の385枚・7撮影元で学習した。
候補枠の形状、画像内の位置、モデル間のIoUとconfidenceから、
最大IoUが0.50〜0.75の6段階のうち何段階を満たすかをリッジ回帰で予測した。
全候補を再選択する方法は品質判定の下限0.22、0.29、0.36を比較し、
表には最高の固定検証値を記載した。補正する方法は、元の候補とIoU 0.50以上で
予測品質が0.05以上高いタイルの枠へ座標を寄せ、対応する元の枠がない場合は
予測品質0.58以上のタイル候補を追加した。全体画像の候補で撮影元を一つずつ
外した品質予測のMAEは0.240〜0.298。4タイルと枠品質を使う方法は固定検証で
比較元を下回ったため、独立確認とKaggle提出の対象から外した。

| 枠品質の下限 | 全体画像で再選択 | 全体画像とタイルで再選択 |
| ---: | ---: | ---: |
| 0.22 | 0.534634 | 0.454414 |
| 0.29 | 0.540310 | 0.482223 |
| 0.36 | 0.544161 | 0.497794 |

再現手順

```bash
# 提出Notebook v1の生成には実行時のcommit 53b276aを使用
uv run --no-sync python scripts/build_submission_notebook.py \
  configs/submission-d2-mbv3-hflip-tta.json --output outputs/proposal-quality/kaggle-tta

# 既存の検出器チェックポイント、固定分割とGWHD 2021の監査済みアーカイブを使用
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --locked --group wheat --with 'ensemble-boxes==1.0.9' \
  python scripts/wheat_proposal_experiment.py field
uv run --locked --group wheat --with 'ensemble-boxes==1.0.9' \
  python scripts/wheat_proposal_experiment.py screen-global

.venv/bin/kaggle kernels output yutarotaro/wheat-benchmark-d2-tiles-validation/1 \
  --file-pattern '^validation-tiles\.json$' -p outputs/proposal-quality
.venv/bin/kaggle kernels output yutarotaro/wheat-benchmark-d2-tiles-field/1 \
  --file-pattern '^field-tiles\.json$' -p outputs/proposal-quality
uv run --locked --group wheat --with 'ensemble-boxes==1.0.9' \
  python scripts/wheat_proposal_experiment.py screen-tiles
```

[TTAの提出Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-infer-d2-mbv3-hflip-tta) ·
[TTAのMLflow推論Run](http://localhost:5050/#/experiments/1/runs/75cbb24154074b9babdcfadec82cdb73) ·
[固定検証のタイル候補Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-benchmark-d2-tiles-validation) ·
[GWHD 2021のタイル候補Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-benchmark-d2-tiles-field) ·
[タイル追加のMLflow Run](http://localhost:5050/#/experiments/1/runs/b533fb9f20c348799ddedf741663a27e) ·
[全体画像の枠品質による再選択のMLflow Run](http://localhost:5050/#/experiments/1/runs/b06619a196ce499cb2c315428dd37a3f) ·
[タイル枠の品質による補正のMLflow Run](http://localhost:5050/#/experiments/1/runs/c8de396d0ab34a8a95fbdb647303aa01) ·
[全体画像とタイルから再選択したMLflow Run](http://localhost:5050/#/experiments/1/runs/0acd815be18d4b8fbce1d06887105b1a)

## 2026-11-15 検出器学習と追加データの比較

目標はKaggle遅延提出のPrivate 0.6000。比較元は提出 `58374318` の
Private 0.5719、固定検証 0.598597、独立確認 0.564296。
次の候補は固定検証636枚で選び、撮影元の悪化が0.01以内か確認する。
採用候補を固定してから独立確認735枚で検証する。

| 検出器 | 実験条件 | 比較元 |
| --- | --- | --- |
| EfficientDet-D2と拡大クロップ | 768px、6 epoch、計3,006更新、元データと監査済みGWHD 2021の385枚。学習サンプルの半分で画像辺の60–82%を切り出す | 同じデータと更新回数で全体画像を使った `1d692cda0f6e4cd2-0002` |
| RT-DETRv2-R18 | COCO事前学習、640px、8 epoch、Kaggleの元データ2,002枚。300クエリで直接枠を予測する | D2単体と既存アンサンブル |
| Faster R-CNN ResNet-50 FPN v2 | COCO事前学習、768px、6 epoch、Kaggleの元データ2,002枚。特徴抽出器とFPNを比較する | MobileNetV3版と既存アンサンブル |

クロップ学習 `01c9cca66fb4432c-0001` は比較用の分割と乱数シード42を使う。
切り出しで元の面積の40%以上が見える枠を境界で切り取り、学習に残す。
最良epochの重みを固定し、epoch別のtrain/valid loss、撮影元別スコア、
小さい穂の再現率、誤検出数をMLflowに保存する。
第1 epochの固定検証は0.475678、全体画像を用いた比較元の同epochは0.472123。
第4 epochでは0.584042、比較元の同epochは0.534023。
[クロップ学習Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-train-d2-6e-768-gwhd385-zoom50-15f7318debe9) ·
[クロップ学習Run](http://localhost:5050/#/experiments/1/runs/c5de427c057d4daaaaded3698e13bca8)

6 epochの学習は完了した。最後のepochは0.574355で、最良は4 epochの
**0.584042**。固定評価のGPUとライブラリの版は比較元と一致した。

| 検出器単体・全体画像で推論 | 全体画像で学習 | 拡大クロップも使って学習 |
| --- | ---: | ---: |
| 固定検証 | 0.547748 | 0.584042 |
| `arvalis_2` | 0.292062 | 0.377470 |
| `rres_1` | 0.668488 | 0.681590 |
| 小さい穂の再現率（IoU 0.50） | 30.51% | 22.05% |
| 誤検出（重複を除く、IoU 0.50） | 4,376 | 3,751 |

クロップ学習の重みをタイル推論でも評価する。学習の打ち切り時期を見る別の比較は、
元のKaggle学習画像だけで12 epochまで実行する `loop.efficientdet-zoomcrop-long.toml`。
6 epochの負例比較で効果を確認した後、12 epochの設定にも拡大クロップ50%と負例5%を入れた（commit `3cbb409`）。

学習投入はNotebookラベルの上限で一度、続いてKaggle APIのHTTP 400で一度失敗した。
後者はNotebook名とVersion 1の両方でHTTP 404を確認して失敗として記録した。
学習に関係しないフォルダを除外し、約660 KiBのNotebookで実行を確認した。
実行時のコミットは `aa2cb62`。

以前のGWHD 2021の385枚で学習したD2も、現行TTAで比較した。既存D2と入れ替えると
固定検証 **0.608373**（`arvalis_2` 0.374217、`rres_1` 0.718947）。
既存2モデルに3台目として加えると **0.608062**（0.373546、0.718805）。
入れ替えを固定して独立確認に進めた。既存手順を提出時のWBF重み
`[1/3, 2/3]` で再集計した基準は0.598586。以前の固定検証では
`[1, 2]` を使い、丸めの影響で報告値から約0.000011差が出た。
比較コードは `98d9e1d`、基準の検証修正は `3d95a8b`。
独立確認は **0.563197**（基準0.564296）。`inrae_1` が0.651745から0.634375に
下がったため、固定検証での改善を再現できず、Kaggle提出の対象から外した。
[固定検証Run](http://localhost:5050/#/experiments/1/runs/3558209b201b4552923b126642ccc059) ·
[独立確認Run](http://localhost:5050/#/experiments/1/runs/0307229f424f406089d2575df3f48e26)

元のKaggle学習画像だけで再学習したD2を3台目に加えた条件は、
固定検証 **0.602975**（`arvalis_2` 0.373280、`rres_1` 0.711442）。
独立確認は **0.573011**（比較元0.564296）で採用条件を満たした。
提出IDは `58377134`、Notebook v1、commit `2edc4ff`。
採点済みの **Public 0.6867 / Private 0.5787** は、前回から0.0095 / 0.0068の改善。
[固定検証Run](http://localhost:5050/#/experiments/1/runs/418c554fa8c64cfdad5f34c96a7ddc09) ·
[独立確認Run](http://localhost:5050/#/experiments/1/runs/3384cb1f9ce04c13af3761fc553569c3) ·
[提出Run](http://localhost:5050/#/experiments/1/runs/abd70768fcfa449f91d4c73795715021) ·
[提出Notebook](https://www.kaggle.com/code/yutarotaro/wheat-infer-mbv3-d2x2-hflip)

現行の3モデルに同じD2重みの1,024px推論を加え、768pxの反転TTAと2:1のWBFで
統合した。固定検証は **0.595007**、`arvalis_2` は0.353744。比較元の0.602975と
0.373280を下回り、独立確認へ進めなかった（commit `213da15`）。

拡大クロップで学習したD2を既存D2と入れ替え、反転TTAで固定検証
**0.620481**（`arvalis_2` 0.427797、`rres_1` 0.711471）。
3台目として足したときは0.615699。入れ替えを固定した独立確認は
**0.587522**（`arvalis_3` 0.561802、`inrae_1` 0.669209）。
同じクロップ学習モデルで4タイル候補を追加した固定検証は0.619134だった。
全体画像の反転TTAで、クロップ学習の効果の大部分が出た。

### 背景クロップを混ぜる学習

拡大クロップを含むD2の学習 `01c9cca66fb4432c-0001` を比較元にする。
[固定学習で採取した413背景クロップ](data-profile.md#追加データと背景サンプルの監査)から毎epochで
異なる100枚を抽出し、2,002サンプルのうち100枚を置き換える。
優先条件は、誤検出の低下と小さい穂の再現率の維持。固定検証に加え、
SPIKEの公開負例10枚で1枚あたりの誤検出数を比較する。
合計3,006更新で比較し、epoch別の背景サンプル数もMLflowに記録する。
再現コマンドの設定は `loop.efficientdet-background.toml`、実装は `fc8eba2`。
負例ありの学習は `07254a28cd6840e2-0001`、実行時commitは `2edc4ff`。
SPIKEの負例10枚では、confidence 0.30で比較元の全体画像学習は平均0.5誤検出/枚、
拡大クロップ学習は平均1.6誤検出/枚だった。0.40では両方とも平均0.1誤検出/枚。
この10枚は補助評価で、固定検証やKaggle Privateのスコアと合算しない。
[負例学習Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-train-d2-6e-768-zoom50-neg05-16980987f167) ·
[負例学習Run](http://localhost:5050/#/experiments/1/runs/5791cb511460484d9521f4e1321cb426)

負例ありは最良5 epochで **0.597398**、6 epochで0.582261。

| 検出器単体・全体画像で推論 | 負例なし | 毎epochに背景100枚 |
| --- | ---: | ---: |
| 固定検証 | 0.584042 | 0.597398 |
| `arvalis_2` | 0.377470 | 0.390293 |
| `rres_1` | 0.681590 | 0.695197 |
| 小さい穂の再現率（IoU 0.50） | 22.05% | 24.66% |
| 誤検出（重複を除く、IoU 0.50） | 3,751 | 2,895 |

SPIKE負例10枚の平均誤検出は、confidence 0.30で1.6から0.5に低下した。
confidence 0.40では0.1から0.3に増えた。少数の背景画像での傾向として記録する。
負例ありのD2を既存D2と入れ替えた反転TTAは、固定検証 **0.623874**、
独立確認 **0.594998**（`arvalis_3` 0.569085、`inrae_1` 0.677301）。
背景追加前の独立確認0.587522から改善した。GWHD 2021の追加学習データを
使った条件で、Kaggle元データのみの12 epoch学習へ手法を移した。
[固定検証Run](http://localhost:5050/#/experiments/1/runs/e36a38d01c684f28b97e08f93815666c) ·
[独立確認Run](http://localhost:5050/#/experiments/1/runs/0cb747db2c594bef93f4ac72894bb73c)

### RT-DETRv2-R18の初回学習

Kaggleの元データだけを使い8 epoch実行した（`66b8fdb7d8944873-0001`、commit `34c041d`）。
固定検証の最良は1 epochの **0.215746**（出力confidence 0.10）。
学習lossは減ったが、8 epochの固定検証は0.170473。保存済み予測のconfidence中央値は0.118、
90%点は0.159で、下限を0.25へ上げると0.002322に低下した。
正解とIoU 0.50を超える候補枠が少なくとも1件ある穂は39.1%。
同じ固定検証での背景入りD2は87.1%だった。今回は候補枠の不足も大きく、
出力の閾値だけでは改善しなかった。
[学習Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-train-rtdetr-v2-r18-8e-640-95dd0119db16)

### Faster R-CNN ResNet-50 FPN v2の初回学習

元データのみ、768px、6 epoch、batch 2、AMPで6,006更新を実行した
（`3f96d2f258b746a8-0001`、commit `34c041d`）。
最良は1 epochの **0.509187**（`arvalis_2` 0.243283、`rres_1` 0.634752）。
6 epochは0.393081。MobileNetV3の基準0.512961を単体では超えなかった。
最良の重みを固定し、現行提出 `58377134` の3モデルに重み0.15で加えた（比較commit `0674326`）。
固定検証は0.602975から **0.604552**、独立確認は0.573011から **0.580625** に上がった。
独立確認の `inrae_1` は0.648325から0.637998へ0.010327低下し、
撮影元別の許容低下0.01を超えたため見送った。
[学習Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-train-frcnn-r50v2-6e-768-7a33d05b8ce3) ·
[学習Run](http://localhost:5050/#/experiments/1/runs/8ea32efd4dda4dc383df4a01de343034) ·
[固定検証Run](http://localhost:5050/#/experiments/1/runs/241cd601149242ecb1f16f8c4d8d567d) ·
[独立確認Run](http://localhost:5050/#/experiments/1/runs/c3d1bd3642e742bc8949c7a2c0f8e2b8)

### Kaggle元データでの12 epoch学習

元データ2,002枚だけで、拡大クロップ50%と毎epochの背景100枚を使ったD2を
12 epoch、6,012更新まで学習した（`06873b044c414c79-0001`、Notebook v1、commit `03707df`）。
元データ6 epochのD2 `1d692cda0f6e4cd2-0001` の最良固定検証は0.539742。
今回の6 epochは0.596102、最良は9 epochの **0.616477**、最終は0.612625だった。
同一学習の6 epoch以降にも改善があり、12 epochまで回して最良重みを選ぶ判断につながった。
元データ6 epochのD2との間では、学習回数に加えてクロップと負例も変わっている。
[学習Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-train-d2-12e-768-zoom50-neg05-945c1bce2c8d) ·
[学習Run](http://localhost:5050/#/experiments/1/runs/b3d63295ff284d2aa777c1b8b66148f3)

反転TTAとWBFを維持したうえで、現行提出 `58377134` を採否の基準にした。
候補の構成と重みは固定検証636枚で比較し、選んだ1構成を独立確認735枚で評価した（比較commit `0674326`）。

| 構成 | 固定検証 | 独立確認 |
| --- | ---: | ---: |
| 現行の提出 `58377134` | 0.602975 | 0.573011 |
| 以前の2モデルに新D2を追加 | 0.618171 | — |
| 現行の3モデルに新D2を追加 | 0.613696 | — |
| 以前の2モデルのD2を新D2に入れ替え | **0.628095** | **0.627904** |

選んだ2モデル構成はFaster R-CNN MobileNetV3と新D2で、WBF重みは1:2。
固定検証の `arvalis_2` は0.373280から0.445026、`rres_1` は0.711442から0.714544に上がった。
独立確認の `arvalis_3` は0.549299から0.596251、`inrae_1` は0.648325から0.728439に上がった。
採用条件を満たしたため、提出推論の設定とNotebookを `e891424` に固定した。
KaggleでNotebook v1の出力10行を検証し、2026-11-16に `58428756` として提出した。
採点済みの **Public 0.6740 / Private 0.5657** は、提出 `58377134` からそれぞれ0.0127 / 0.0130低下した。
固定検証は+0.025120、独立確認は+0.054893だった。今回の比較では、欧州4撮影元で
観測した改善がKaggleのテストで再現しなかった。次は異なる撮影条件の公開データで
両提出の予測数と枠の一致を調べ、どの条件で差が生じるか確認する。
[固定検証Run](http://localhost:5050/#/experiments/1/runs/0c651e069ef9407b9fcd80fa4d9ab0b5) ·
[独立確認Run](http://localhost:5050/#/experiments/1/runs/94b31b90d3d84c1f99c9bc7b232eb601) ·
[提出Run](http://localhost:5050/#/experiments/1/runs/1d00d294543a4a3d9c1f22baf34dc8a0) ·
[提出推論Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-infer-d2-12e-neg05-06873b04)

次の検出器候補は、Kaggle元データ、拡大クロップ50%、背景5%を使う1,024pxの
EfficientDet-D4。D2で小さい穂の再現率が16.51%だったため、入力解像度を上げた状態で
学習して小さい穂の再現率と固定検証を比べる。D4の6 epochは6,006更新を予定し、
D2の6 epoch（3,006更新、0.596102）と最終12 epoch（6,012更新、0.612625）を参照する。
BiFPNの深さやバッチサイズも異なるので、モデルと学習条件を合わせた比較になる。
ローカルの学習・推論smokeの後、`71e04a45512846e2-0001` をKaggleに投入した（学習commit `c66cca4`）。
実行設定は `loop.efficientdet-d4.toml`。
[D4学習Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-train-d4-6e-1024-zoom50-neg05-6d3922b424ab) ·
[D4学習Run](http://localhost:5050/#/experiments/1/runs/78168ed43a0543b396419a92abd97be8)

### 異なる撮影条件での提出構成の比較

提出 `58377134` と `58428756` の重み、TTA、WBF、confidenceを固定し、
同じ提出推論コードをSPIKEの公開テスト30枚とSD-RWHDの159枚で実行した（commit `e92b3e6`）。
SPIKEはオーストラリアの斜め撮影、SD-RWHDは米国の撮影で、後者は枠の面積が画像の0.2%未満の穂が61.3%を占める。
公表された注釈を使った補助評価として、データセットごとに結果を集計した。

| 公開データ | 測定項目 | 3モデル `58377134` | 新D2との2モデル `58428756` |
| --- | --- | ---: | ---: |
| SPIKE 30枚 | Wheat Image Precision | 0.064765 | 0.073638 |
| SPIKE 30枚 | 一致した枠 / 予測枠、IoU 0.50 | 331 / 908 | 398 / 1,031 |
| SD-RWHD 159枚 | Wheat Image Precision | 0.067430 | 0.045767 |
| SD-RWHD 159枚 | 一致した枠 / 予測枠、IoU 0.50 | 1,954 / 3,831 | 1,424 / 2,776 |
| SD-RWHD 159枚 | 小さい穂の検出数 / 注釈数、IoU 0.50 | 285 / 11,516 | 182 / 11,516 |

新構成の影響は撮影条件で異なった。SD-RWHDでは予測枠と一致した枠がともに減り、
検出できた小さい穂は2.47%から1.58%に低下した。D4は固定検証の撮影元別に加え、
同じ公開データで小さい穂の再現率を比較する。
SPIKEは穂の注釈漏れの指摘があるため、元の注釈での値として保存する。
[MLflowの比較Runと撮影条件別集計](http://localhost:5050/#/experiments/1/runs/9a8aa3a67d65411994612b2fa937c0e4) ·
[公開データの準備と監査](data-profile.md#追加データと背景サンプルの監査)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --locked --group wheat --group detr --group dev --group kaggle \
  --with ensemble-boxes==1.0.9 \
  python scripts/compare_external_fields.py configs/submission-mbv3-d2x2-hflip.json \
  configs/submission-d2-12e-neg05-06873b04.json --output outputs/model-search/external-fields.json
```

## 2026-11-16 実験方針の再評価

目標はKaggle Private 0.6000。現在の最高は `58377134` の0.5787。`58428756` は
固定検証で+0.025120、独立確認で+0.054893だったが、Kaggle Privateが0.0130低下した。
国や撮影条件で改善の向きが変わり、SD-RWHDでは小さい穂の検出率が2.47%から1.58%へ下がった。
固定検証は636枚のうち432枚が `rres_1`。`arvalis_2` は穂25本未満の画像が76.5%で、
面積比0.2%未満の穂は1.2%。探索時は全体、撮影元別、小さい穂の再現率を併記する。
大会運営者の分析でも、小さい穂と撮影セッションによって見逃しが増えていた。
[大会運営者の分析](https://pmc.ncbi.nlm.nih.gov/articles/PMC10795497/)

| 優先順 | 仮説と比較 | 次の判断 |
| --- | --- | --- |
| 1 | D4の2〜3 epoch目の崩れに、batch 2で更新する事前学習済みバックボーンのBatchNorm統計が関係する。元のD4と同じ順序・設定で統計の更新だけを止め、最初の3 epochを比べる。 | 2・3 epoch目の固定検証がともに0.50以上で、最良値が元の最初の3 epochの最良0.582839から0.01以内なら、長い学習を検討する。小さい穂の再現率と撮影元別の差も記録する。 |
| 2 | 小さい穂の不足を、近くから撮ったような拡大だけで補うのは難しい。元データの縮小方向も使う学習を、同じ検出器・更新回数で比較する。 | 固定検証の小さい穂の再現率と誤検出数を比較し、撮影条件の違う公開データでも確認する。既存タイル推論の再試行だけで予算を使わない。 |
| 3 | 追加データは画像数以上に、元データと異なる撮影条件・穂の大きさ・注釈品質が効く。SPIKEの学習候補290枚の注釈を確認し、元データだけの対照と更新回数をそろえて比較する。 | 同一分割での改善と、データの利用条件を確認してから提出候補にする。SPIKEの公開テスト30枚とSD-RWHDは同じ追加データ実験の学習から外す。 |

D4の固定検証は1〜4 epochで0.582839、0.089448、0.024386、0.579102。
誤検出数は3,381、14,595、14、1,416で、小さい穂の再現率は18.70%、1.04%、0%、8.57%。
検証lossは0.48716、0.97469、0.87668、0.53588。検証時はBatchNormが評価モードになり、
学習時は統計が更新される。元のEfficientDetは同期BatchNormを使う。現在のログでは原因を特定できないため、
既存D4の完了と重みの回収を続け、単独の変更で仮説を検証する。
対照実験の設定は [`loop.efficientdet-d4-frozenbn.toml`](../loop.efficientdet-d4-frozenbn.toml)。
[D4の学習履歴](http://localhost:5050/#/experiments/1/runs/78168ed43a0543b396419a92abd97be8) ·
[EfficientDet原論文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.pdf)

元のD4は6 epochを完了し、別の評価プロセスで同じ固定検証636枚を採点した結果も、最良の6 epochで **0.594707** だった。
`arvalis_2` は0.414744、`rres_1` は0.679690、小さい穂の再現率は12.12%。
1 epoch目の18.70%より低く、5 epoch目の固定検証は0.026441だった。
同じseedでサンプラーを再現すると、背景だけのバッチは最初の5 epochで2、5、2、1、1件だった。
バッチの件数だけでは成績の崩れと対応しない。ログの `optimizer_steps` は処理したバッチの累計で、
AMPが更新を省略した回数は計測していない。

対照試行 `77baee4323ce4da0-0001` を投入した（Notebook v1、学習commit `4772447`）。
バックボーンのBatchNorm統計を固定し、学習可能な重みと検出ヘッド側のBatchNormは通常どおり更新する。
[対照試行の学習Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-train-d4-3e-1024-frozenbn-36c240073dcf) ·
[対照試行のMLflow Run](http://localhost:5050/#/experiments/1/runs/47d191ae75784abfbe3e4a37acdf1b48)

探索には既存の固定検証636枚を使用する。候補の構成と閾値を固定してから、既存の独立確認735枚を使う。
この735枚には過去の採用判断で使った履歴がある。公開データは補助診断として扱い、
Kaggle採点は目標到達と提出後の記録に使う。新しい仮説は、先に比較元・変更・測定項目・採否を記録し、
結果と見送った理由を実験系譜に追記する。

実験を始めたら、以下を記録する。

| 項目 | 内容 |
| --- | --- |
| 実験ID・日時 | 重複しないID、UTCのタイムゾーン付き時刻 |
| 目的・仮説 | 何を変え、何が改善すると考えたか |
| コード | コミットSHAと未コミット差分の有無 |
| データ・分割 | データの版、保存した分割ファイル |
| 設定 | モデル、画像サイズ、バッチ数、学習回数、シード |
| 環境 | Python・ライブラリのバージョン、GPU |
| 結果 | コンペ方式の検証スコア、実行時間 |
| 成果物 | 重み・ログ・予測画像の保存先、Kaggle実行のURLと版 |
| 考察 | 見逃し・誤検出・重複・枠のずれ、次に試すこと |

比較する実験では検証分割と評価方法を固定する。
