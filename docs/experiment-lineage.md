# 小麦検出の実験系譜

固定検証636枚を使った学習と、その重み・予測を使った推論を載せる。学習図の実線は
親のソース・設定からの派生、点線は別セッションでの比較や再試行を示す。
推論と提出の図では、実線が重み・予測の入力を示す。試行の仮説と集計の詳細は
[実験記録](experiments.md) と [データ分布](data-profile.md) を参照。

## 学習の分岐

スコアは全試行で共通の Wheat Image Precision、confidence 0.30。学習が完了する前に
止まった実行には停止理由を示す。

```mermaid
flowchart TB
    B3["Faster R-CNN MBv3 · 3 epoch<br/>基準モデルで検証と提出を確かめる<br/>0.445099 · base dbb8fb9 †"]
    BG["背景分類損失 1.25倍<br/>偽陽性を抑える<br/>0.451619 · base dbb8fb9 †"]
    BOX["枠回帰損失 1.20倍<br/>枠の位置ずれを減らす<br/>0.450926 · base dbb8fb9 †"]
    POS["正例分類損失 1.10倍<br/>見逃しを減らす<br/>0.439911 · base dbb8fb9 †"]
    WARM["背景損失のウォームアップ<br/>偽陽性抑制と再現率を両立する<br/>0.448220 · base dbb8fb9 †"]
    B12["Faster R-CNN MBv3 · 12 epoch<br/>学習を延ばして検証の推移を測る<br/>0.512961 · commit 14f596b"]
    COLOR["色ジッター 0.15<br/>撮影元ごとの見え方に対応する<br/>0.509165 · base 14f596b †"]
    D2["EfficientDet-D2 · 768px · 6 epoch<br/>高解像度で小さい穂を拾う<br/>0.526891 · commit 960cc1b"]
    MOS["D2 ＋ Mosaic 0.50<br/>背景と穂密度の組み合わせを増やす<br/>0.512726 · commit eff9cf1 ‡"]
    MON1["D2 ＋ Mons 60枚 · 初回<br/>穂の少ない撮影条件を補う<br/>未評価 · 図の同梱で停止 · 53c0cc2"]
    MON2["D2 ＋ Mons 60枚 · 再試行<br/>文書中の図を同梱対象から外す<br/>未評価 · キュー中に期限切れ · f51aeb6"]
    MON3["D2 ＋ Mons 60枚 · 6時間予算<br/>キュー待ち込みの期限を延ばす<br/>未評価 · ZIPを検出できず停止 · 1e5d4c2"]
    Y["YOLOX-S · 768px · 6 epoch<br/>異なる検出器でD2の見逃しを補えるか<br/>0.345790 · commit 3cbfdcb"]

    B3 --> BG
    BG --> BOX
    BG --> POS
    BG --> WARM
    B3 -. 比較 .-> B12
    B12 --> COLOR
    B12 -. 比較 .-> D2
    D2 --> MOS
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
`base` はGitの基点で、各試行のソースは末尾のハッシュで識別する。
‡ `eff9cf1` は学習前に固定した試行コミットで、ローカルの
`refs/ml-loop/trials/d31623fdd5114e49-0002` にある。

## アンサンブルへの合流

WBFの枠統合IoUは0.55。枠の統合で誤検出や重複が減るかを調べた。YOLOX-Sの追加候補は
出力confidence 0.40で既存のD2＋Faster R-CNNと比較した。

```mermaid
flowchart TB
    C["Faster R-CNN ＋ 色ジッター<br/>撮影元への頑健性 · base 14f596b †"]
    B12["Faster R-CNN · 12 epoch<br/>長めの学習 · commit 14f596b"]
    D2["EfficientDet-D2<br/>高解像度 · commit 960cc1b"]
    Y["YOLOX-S<br/>見逃しの補完候補 · commit 3cbfdcb"]

    W1["横反転＋色ジッターのWBF<br/>重みの違いによる誤検出減を検証<br/>均等 0.532910 / 2対1 0.532010 · conf 0.50<br/>比較元は各単体 · commit 29b9e6d"]
    W2["D2＋Faster R-CNNのWBF<br/>異なる検出器で枠を補正<br/>0.569306 at 0.50 / 0.580485 at 0.40<br/>commit 354401d"]
    W3["D2＋YOLOX-S＋Faster R-CNNのWBF<br/>既存2モデルにYOLOX-Sの予測を追加<br/>0.557939 · conf 0.40 · commit dcb26bd<br/>比較元は既存WBF 0.580485"]
    WY["D2＋YOLOX-SのWBF<br/>別の検出器で見逃しを補う<br/>0.477607 · conf 0.40 · commit 22ab781<br/>比較元はD2単体と既存WBF"]

    C --> W1
    B12 --> W1
    B12 --> W2
    D2 --> W2
    B12 --> W3
    D2 --> W3
    Y --> W3
    D2 --> WY
    Y --> WY

    classDef ensemble fill:#eee9fc,color:#392863,stroke:#8d76b3
    class W1,W2,WY,W3 ensemble
```

YOLOX-S単体はconfidence 0.30で小さい穂の再現率19.44%（IoU 0.50）だった。
同じconfidenceの誤検出6,521件と見逃し6,478件を踏まえ、総合スコアは0.345790。
既存WBFに加えた3モデル統合も全体で0.022546低く、撮影元別でも両方が低下した。
比較基準はD2＋Faster R-CNNのWBFを使う。

- [YOLOX-S 学習](http://localhost:5050/#/experiments/1/runs/4187c38c343e49c3a7dfdd9fd66dd85f) · [学習Notebook v1](https://www.kaggle.com/code/yutarotaro/wheat-train-yolox-s-6e-768-hflip-3c34d8001e01)
- [D2＋Faster R-CNN](http://localhost:5050/#/experiments/1/runs/3600138dbd634b52b2e050d204d9ee51) · [D2＋YOLOX-S](http://localhost:5050/#/experiments/1/runs/f7f52f79d74a41379b67bbc10e661c7c) · [3モデル](http://localhost:5050/#/experiments/1/runs/d5ffac91d7a1458eaf683447dd969548)

## Kaggleへの提出

```mermaid
flowchart TB
    B3["Faster R-CNN · 3 epoch<br/>固定検証 0.445099 · base dbb8fb9 †"]
    W2["D2＋Faster R-CNNのWBF<br/>固定検証 0.580485 · commit 354401d"]
    Y["YOLOX-S単体<br/>固定検証 0.345790 · commit 3cbfdcb"]
    S1["Kaggle 58331537 · v1<br/>初期単体モデルの提出を確認<br/>Public 0.5358 / Private 0.4449<br/>Notebook commit 14f596b"]
    S2["Kaggle 58342094 · v1<br/>固定検証の最良候補を提出<br/>Public 0.6512 / Private 0.5466<br/>比較元 58331537 · commit 98308fd"]
    SY["Kaggle 58362730 · YOLOX-S単体 · v1<br/>単体のKaggle基準を測定<br/>Public 0.5645 / Private 0.5005<br/>比較元 58331537 · commit dcb26bd"]

    B3 --> S1
    W2 --> S2
    Y --> SY

    classDef submission fill:#e6f3eb,color:#204d34,stroke:#73a287
    class S1,S2,SY submission
```

- [初期単体の提出Notebook](https://www.kaggle.com/code/yutarotaro/wheat-baseline-late-20261114) · [D2＋Faster R-CNNの提出Notebook](https://www.kaggle.com/code/yutarotaro/wheat-infer-d2-mbv3-wbf040) · [YOLOX-S単体の提出Notebook](https://www.kaggle.com/code/yutarotaro/wheat-infer-yolox-s-6e-768-conf030)

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
