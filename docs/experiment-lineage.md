# 小麦検出の実験系譜

固定検証636枚を使った学習と、その重み・予測を使った推論を載せる。学習図の実線は
親のソース・設定からの派生、点線は別セッションでの比較や再試行を示す。
推論と提出の図では、実線が重み・予測の入力を示す。試行の仮説と集計の詳細は
[実験記録](experiments.md) と [データ分布](data-profile.md) を参照。

## 学習の分岐

スコアは全試行で共通の Wheat Image Precision、confidence 0.30。学習が完了する前に
止まった実行には停止理由を示す。

```mermaid
%%{init: {"htmlLabels": false, "flowchart": {"padding": 24, "nodeSpacing": 28}}}%%
flowchart TB
    B3["Faster R-CNN MBv3<br/>3 epoch · 検証と提出の基準<br/>0.445099<br/>base dbb8fb9 †"]
    BG["背景分類損失 1.25倍<br/>偽陽性を抑える<br/>0.451619<br/>base dbb8fb9 †"]
    BOX["枠回帰損失 1.20倍<br/>枠の位置ずれを減らす<br/>0.450926<br/>base dbb8fb9 †"]
    POS["正例分類損失 1.10倍<br/>見逃しを減らす<br/>0.439911<br/>base dbb8fb9 †"]
    WARM["背景損失<br/>ウォームアップ<br/>誤検出と再現率を調整<br/>0.448220<br/>base dbb8fb9 †"]
    B12["Faster R-CNN MBv3<br/>12 epoch · 検証の推移を測る<br/>0.512961<br/>commit 14f596b"]
    COLOR["色ジッター 0.15<br/>撮影元ごとの見え方に対応<br/>0.509165<br/>base 14f596b †"]
    D2["EfficientDet-D2<br/>768px · 6 epoch<br/>高解像度で小さい穂を拾う<br/>0.526891<br/>commit 960cc1b"]
    MOS["D2 ＋ Mosaic 0.50<br/>背景と穂密度を組み替える<br/>0.512726<br/>commit eff9cf1 ‡"]
    EPO["D2 · 各epochを採点<br/>最良時点を選ぶ基準<br/>0.539742 · epoch 3<br/>commit 83c9c31"]
    F385["D2 ＋ 新規385枚<br/>更新回数をそろえて比較<br/>0.547748 · epoch 6<br/>commit 77083de ‡"]
    MON1["D2 ＋ Mons 60枚 · 初回<br/>穂の少ない条件を補う<br/>未評価 · 図の同梱で停止<br/>commit 53c0cc2"]
    MON2["D2 ＋ Mons 60枚 · 再試行<br/>文書の図を同梱から除外<br/>未評価 · キュー中に期限切れ<br/>commit f51aeb6"]
    MON3["D2 ＋ Mons 60枚<br/>待ち時間込みの予算を6時間へ<br/>未評価 · ZIPを検出できず停止<br/>commit 1e5d4c2"]
    Y["YOLOX-S · 768px · 6 epoch<br/>D2の見逃しを補えるか<br/>0.345790<br/>commit 3cbfdcb"]

    B3 --> BG
    BG --> BOX
    BG --> POS
    BG --> WARM
    B3 -. 比較 .-> B12
    B12 --> COLOR
    B12 -. 比較 .-> D2
    D2 --> MOS
    D2 -. 比較 .-> EPO
    EPO --> F385
    D2 -. 比較 .-> MON1
    MON1 -. 再試行 .-> MON2
    MON2 -. 再試行 .-> MON3
    D2 -. 比較 .-> Y

    classDef reference fill:#e0edff,color:#18365e,stroke:#6685b0
    classDef unscored fill:#fff0e8,color:#633419,stroke:#bb8466
    class B3,B12,D2,Y reference
    class MON1,MON2,MON3 unscored
```

† 初期の一部試行はベースコミットに凍結済みの作業差分を加えて実行した。図の
`base` はGitの基点で、各試行のソースは末尾のハッシュで識別する。MLflowの
`experiment.ipynb` から凍結済みコードを読める。元がKaggle Scriptの試行には、
元の投入ファイルからコードを展開した閲覧用Notebookを保存した。
‡ 学習前に固定した試行コミット。ローカルの `refs/ml-loop/trials/` に加え、
GitHubの `codex/wheat-trials/<試行ID>` にも保存した。
`eff9cf1` は `d31623fdd5114e49-0002`、`77083de` は `1d692cda0f6e4cd2-0002` に対応する。

## アンサンブルへの合流

WBFの枠統合IoUは0.55。枠の統合で誤検出や重複が減るかを調べた。YOLOX-Sの追加候補は
出力confidence 0.40で既存のD2＋Faster R-CNNと比較した。

```mermaid
%%{init: {"htmlLabels": false, "flowchart": {"padding": 24, "nodeSpacing": 28}}}%%
flowchart TB
    C["Faster R-CNN ＋ 色ジッター<br/>撮影元への頑健性 · base 14f596b †"]
    B12["Faster R-CNN · 12 epoch<br/>長めの学習 · commit 14f596b"]
    D2["EfficientDet-D2<br/>高解像度 · commit 960cc1b"]
    F385["D2 ＋ 新規385枚<br/>小さい穂を補う<br/>commit 77083de ‡"]
    Y["YOLOX-S<br/>見逃しの補完候補 · commit 3cbfdcb"]

    W1["横反転＋色ジッターのWBF<br/>重みを変えて誤検出を比較<br/>conf 0.50<br/>均等 0.532910<br/>2対1 0.532010<br/>比較元は各単体<br/>commit 29b9e6d"]
    W2["D2＋Faster R-CNNのWBF<br/>異なる検出器で枠を補正<br/>conf 0.50 · 0.569306<br/>conf 0.40 · 0.580485<br/>commit 354401d"]
    W4["既存D2を2倍にするWBF<br/>IoU 0.45 · conf 0.40<br/>0.587241 · 比較元 0.580485<br/>commit 83c9c31"]
    WF["追加D2＋Faster R-CNN<br/>均等WBF · 0.586916<br/>比較元は既存WBF 0.580485<br/>commit 64ba330"]
    W3["3モデルのWBF<br/>既存WBFにYOLOX-Sを追加<br/>conf 0.40 · 0.557939<br/>比較元は既存WBF 0.580485<br/>commit dcb26bd"]
    WY["D2＋YOLOX-SのWBF<br/>別の検出器で見逃しを補う<br/>conf 0.40 · 0.477607<br/>比較元はD2単体と既存WBF<br/>commit 22ab781"]

    C --> W1
    B12 --> W1
    B12 --> W2
    D2 --> W2
    B12 --> W4
    D2 --> W4
    B12 --> WF
    F385 --> WF
    B12 --> W3
    D2 --> W3
    Y --> W3
    D2 --> WY
    Y --> WY

    classDef ensemble fill:#eee9fc,color:#392863,stroke:#8d76b3
    class W1,W2,WY,W3,W4,WF ensemble
```

YOLOX-S単体はconfidence 0.30で小さい穂の再現率19.44%（IoU 0.50）だった。
同じconfidenceの誤検出6,521件と見逃し6,478件を踏まえ、総合スコアは0.345790。
既存WBFに加えた3モデル統合も全体で0.022546低く、撮影元別でも両方が低下した。
比較基準はD2＋Faster R-CNNのWBFを使う。

- [YOLOX-S 学習](http://localhost:5050/#/experiments/1/runs/4187c38c343e49c3a7dfdd9fd66dd85f) · [学習Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-train-yolox-s-6e-768-hflip-3c34d8001e01)
- [D2＋Faster R-CNN](http://localhost:5050/#/experiments/1/runs/3600138dbd634b52b2e050d204d9ee51) · [D2＋YOLOX-S](http://localhost:5050/#/experiments/1/runs/f7f52f79d74a41379b67bbc10e661c7c) · [3モデル](http://localhost:5050/#/experiments/1/runs/d5ffac91d7a1458eaf683447dd969548)

## Kaggleへの提出

```mermaid
%%{init: {"htmlLabels": false, "flowchart": {"padding": 24, "nodeSpacing": 28}}}%%
flowchart TB
    B3["Faster R-CNN · 3 epoch<br/>固定検証 0.445099<br/>base dbb8fb9 †"]
    W2["D2＋Faster R-CNNのWBF<br/>固定検証 0.580485<br/>commit 354401d"]
    W4["既存D2を2倍にするWBF<br/>固定検証 0.587241<br/>最終確認 0.557030<br/>commit 83c9c31"]
    Y["YOLOX-S単体<br/>固定検証 0.345790<br/>commit 3cbfdcb"]
    S1["Kaggle 58331537 · v1<br/>初期単体モデルの提出<br/>Public 0.5358<br/>Private 0.4449<br/>Notebook commit 14f596b"]
    S2["Kaggle 58342094 · v1<br/>固定検証の最良候補を提出<br/>Public 0.6512<br/>Private 0.5466<br/>比較元 58331537<br/>commit 98308fd"]
    SY["Kaggle 58362730 · v1<br/>YOLOX-S単体のKaggle基準<br/>Public 0.5645<br/>Private 0.5005<br/>比較元 58331537<br/>commit dcb26bd"]
    S4["Kaggle 58370638 · v1<br/>WBFの調整を確認<br/>Public 0.6668<br/>Private 0.5651<br/>比較元 58342094<br/>commit 64ba330"]

    B3 --> S1
    W2 --> S2
    W4 --> S4
    Y --> SY

    classDef submission fill:#e6f3eb,color:#204d34,stroke:#73a287
    class S1,S2,SY,S4 submission
```

- [初期単体の提出Notebook](https://www.kaggle.com/code/yutarotaro/wheat-baseline-late-20261114) · [D2＋Faster R-CNNの提出Notebook](https://www.kaggle.com/code/yutarotaro/wheat-infer-d2-mbv3-wbf040) · [YOLOX-S単体の提出Notebook](https://www.kaggle.com/code/yutarotaro/wheat-infer-yolox-s-6e-768-conf030)
- [Private 0.5651の提出Notebook](https://www.kaggle.com/code/yutarotaro/wheat-infer-d2-mbv3-wbf045-d2x2) · [MLflow提出Run](http://localhost:5050/#/experiments/1/runs/de5bd96bb3964629938b13ec172dd35a)

## 画像全体・反転・タイルの比較

```mermaid
%%{init: {"htmlLabels": false, "flowchart": {"padding": 24, "nodeSpacing": 28}}}%%
flowchart TB
    W["提出 58370638のWBF<br/>固定検証 0.587241<br/>独立確認 0.557030<br/>commit 64ba330"]
    T["水平反転した枠を統合<br/>反転で見逃しを補う<br/>固定検証 0.598597<br/>独立確認 0.564296<br/>commit 53b276a"]
    P["D2の4タイル候補を追加<br/>小さい穂の見逃しを補う<br/>固定検証 0.572241<br/>commit fc86e28"]
    Q["学習した枠品質で再選択<br/>枠の重なりを推定する<br/>固定検証 0.544161<br/>commit 569217c"]
    R["枠品質で座標補正と候補追加<br/>タイル由来の誤検出を抑える<br/>固定検証 0.558535<br/>実装 commit c4d3328"]
    A["全体画像とタイルから再選択<br/>枠品質で順位を付け直す<br/>固定検証 0.497794<br/>実装 commit c4d3328"]
    S["Kaggle 58374318 · v1<br/>Public 0.6772<br/>Private 0.5719<br/>比較元 58370638<br/>commit 53b276a"]

    W --> T --> S
    W --> P
    W --> Q
    P --> R
    Q --> R
    P --> A
    Q --> A

    classDef submission fill:#e6f3eb,color:#204d34,stroke:#73a287
    class S submission
```

- [水平反転TTAの提出Notebook](https://www.kaggle.com/code/yutarotaro/wheat-infer-d2-mbv3-hflip-tta) · [MLflow提出Run](http://localhost:5050/#/experiments/1/runs/75cbb24154074b9babdcfadec82cdb73)
- MLflowの推論Runの親は表示用に1件を選ぶ。アンサンブルで使った全学習RunはDescriptionと `model_input_run_ids` で確認できる。

## 拡大クロップと背景を使う検出器学習

```mermaid
%%{init: {"htmlLabels": false, "flowchart": {"padding": 24, "nodeSpacing": 28}}}%%
flowchart TB
    M["Faster R-CNN MBv3<br/>既存の全体画像モデル<br/>固定検証 0.512961<br/>base 14f596b"]
    D["D2 · 元データのみ<br/>最良epochを保存<br/>固定検証 0.539742<br/>commit 83c9c31"]
    F["D2 · 追加385枚<br/>撮影元を増やす<br/>固定検証 0.547748<br/>commit 77083de"]
    C["D2 · 拡大クロップ50%<br/>画角を変えて学習<br/>固定検証 0.584042<br/>commit aa2cb62"]
    N["D2 · 背景クロップ5%<br/>誤検出の低下を比較<br/>固定検証 0.597398<br/>commit 2edc4ff"]
    U["負例ありD2に入れ替え<br/>同じ反転TTAで比較<br/>固定検証 0.623874<br/>独立確認 0.594998<br/>学習 commit 2edc4ff"]
    S["3モデルの反転TTA<br/>固定検証 0.602975<br/>独立確認 0.573011<br/>Public 0.6867<br/>Private 0.5787<br/>提出 58377134 · v1<br/>commit 2edc4ff"]
    Z["拡大クロップD2に入れ替え<br/>撮影元をまたぐ改善を確認<br/>固定検証 0.620481<br/>独立確認 0.587522<br/>学習 commit aa2cb62"]
    T["RT-DETRv2-R18<br/>直接枠を予測する方式<br/>固定検証 0.215746<br/>commit 34c041d"]
    R["Faster R-CNN R50 v2<br/>FPNと特徴抽出器を比較<br/>固定検証 0.509187<br/>commit 34c041d"]
    R15["R50を重み0.15で追加<br/>固定検証 0.604552<br/>独立確認 0.580625<br/>圃場別基準で見送り<br/>比較 commit 0674326"]
    L["D2 · 12 epoch<br/>元データでクロップと負例<br/>固定検証 0.616477 · epoch 9<br/>commit 03707df"]
    P["D2を入れ替えて反転TTA<br/>固定検証 0.628095<br/>独立確認 0.627904<br/>Public 0.6740<br/>Private 0.5657<br/>提出 58428756 · v1<br/>提出推論 commit e891424"]
    D4["D4 · 1,024px · 6 epoch<br/>小さい穂の再現率を比較<br/>学習投入済み<br/>commit c66cca4"]

    D --> F --> C --> N
    C --> Z
    M --> Z
    D --> S
    M --> S
    D --> T
    D --> L
    N --> L
    N --> U
    M --> U
    M --> R
    R --> R15
    S --> R15
    L --> P
    M --> P
    S -. 比較 .-> P
    L -. 比較 .-> D4
```

<details>
<summary>初期試行のIDと凍結ソースハッシュ</summary>

| 試行 | ID | ソースSHA-256の先頭12桁 |
| --- | --- | --- |
| 3 epoch基準 | `10a1829d2fc84c67-0001` | `d0dfd016882d` |
| 背景分類損失 | `f5d6ac6937e943b8-0002` | `49e17622b751` |
| 枠回帰損失 | `f5d6ac6937e943b8-0003` | `e16db553fb66` |
| 正例分類損失 | `f5d6ac6937e943b8-0004` | `1b64af912199` |
| 背景損失のウォームアップ | `f5d6ac6937e943b8-0005` | `792f7c799877` |
| 色ジッター | `8fc98f3e68e8441a-0002` | `00e897919792` |

</details>
