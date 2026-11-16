# Global Wheat Detection

小麦の穂をバウンディングボックスで検出する、Kaggleの過去コンペに取り組む。

- コンペ: https://www.kaggle.com/competitions/global-wheat-detection
- コード・設定・実験記録の本体はこのローカルリポジトリ。
- GitHubのprivateリポジトリでコードと記録を管理する。
- データ確認と可視化はMac、本格的な学習はKaggle GPUで行う。

## 現在の段階

[ML Research Loop](https://github.com/Yutarotaro/ml-research-loop)に接続済み。
固定分割、採点、GPU向け学習・推論、独立評価によるエラー画像出力を実装した。
Kaggle公式データでGPUベースラインと短時間の改善試行を実行した。
再現条件とスコアは [実験記録](docs/experiments.md)、分岐と合流は [実験系譜](docs/experiment-lineage.md) を参照。
撮影元ごとの画像枚数・穂数・枠サイズ・明るさは [データ分布の集計](docs/data-profile.md) を参照。
提出 `58342094` までの手法・モデル構成図・評価結果は [技術論文ドラフト](docs/paper-58342094.md) を参照。

## 環境

Python 3.12を使用する。依存パッケージの解決結果は`uv.lock`に保存する。

```bash
uv sync --locked --group kaggle --group wheat
```

標準のuvキャッシュに書き込めない環境では、コマンドの前に
`UV_CACHE_DIR=.cache/uv`を付けてプロジェクト内のキャッシュを使う。

Kaggle CLIの確認:

```bash
uv run --locked --group kaggle kaggle --version
```

Kaggle CLI 2.2.5の依存宣言に含まれていない`six`も、起動に必要なため明示的に追加している。

## ディレクトリ

```text
data/          # コンペの入力データ（Git対象外）
models/        # 学習済み重み（Git対象外）
outputs/       # 可視化、ログ、予測、提出ファイル（Git対象外）
src/           # 共通のPython処理を今後追加
scripts/       # データ確認・評価などの実行入口を今後追加
configs/       # 実験設定を今後追加
kaggle/        # Kaggle実行用コード・メタデータを今後追加
docs/          # 設計と実験記録
```

画像・重み・認証情報はGitに入れない。Notebookを追加する場合は、コミット前に出力を除去する。
`.gitignore`はNotebookの埋め込み画像やコードに直接書いた秘密情報を除去しないため、差分も確認する。

## データ取得前の準備

現在のMacではAPI認証を設定済み。別の環境で再現する場合は以下を行う。

1. Kaggleでコンペの規約を読み、本人が同意する。
2. Kaggleのアカウント設定でAPI認証を設定する。トークンはチャットやリポジトリに貼らない。
3. CLIで認証とコンペデータへのアクセスを確認する。

準備後のダウンロード先は`data/raw/`とする。

```bash
uv run --locked --group kaggle kaggle competitions download \
  -c global-wheat-detection -p data/raw
```

## 進め方

1. 画像と正解枠を可視化し、データの形式・欠損・不正な枠・撮影元を確認する。
2. 検証分割を設計し、分割に使う画像IDを保存する。撮影元の違いを考慮する。
3. コンペ固有の評価処理を実装し、見逃し・誤検出・重複検出などの例で確認する。
4. 事前学習済みモデルで小規模な学習・推論・評価を通す。
5. Kaggle GPUへ学習を投入し、重み・設定・ログ・予測結果を保存する。
6. 同じ検証分割で改善実験を比較する。

乱数シードの初期値は`42`。時刻はUTCのタイムゾーン付きで記録する。
コンペの評価方法は一般的なCOCO mAPと異なるため、ライブラリ標準値で代用しない。

## Kaggleでの学習と自走

Kaggleのデータ利用条件を確認し、`data/raw/`直下に`train.csv`と`train/*.jpg`を展開する。
先に固定分割を作成する。同じ撮影元や同一画像を別の分割に置かない。
`data/prepared/split.json`には画像ハッシュと分割情報が含まれる。共有・Git管理はしない。

```bash
uv sync --locked --group kaggle --group wheat
uv run --locked --group kaggle --group wheat python src/wheat/data.py
uv run --locked --group kaggle --group wheat python src/wheat/train.py --smoke
```

Loopリポジトリ側で`uv sync --locked --no-editable`を実行する。
未コミットのプロジェクト変更を整理し、次を小麦リポジトリから実行する。

```bash
../ml-research-loop/.venv/bin/ml-loop doctor --project .
../ml-research-loop/.venv/bin/ml-loop run --project . --state .loop-wheat --budget 1h --trials 1
```

実行ディレクトリの`report`でベースラインのスコアを確認する。
次に、得られた実行ディレクトリを`--baseline`へ渡す。

```bash
../ml-research-loop/.venv/bin/ml-loop run --project . --state .loop-wheat --budget 4h --baseline BASELINE_RUN
```

現在の `loop.toml` では、12 epochの基準試行で観測したvalid lossを踏まえ、
6 epoch・640px・横反転を基準に、色補正と上下反転をそれぞれ試す。
固定検証の `wheat_image_precision` の目標は `0.90`。1試行最大45分で、比較単位をそろえてから次の候補を試す。

EfficientDetとMosaicを比較する場合は、同じ固定検証で次を実行する。
EfficientDet-D2を768pxで6 epoch学習し、横反転のみと、Mosaicを確率0.50で追加した条件を順に比較する。

```bash
uv sync --locked --group kaggle --group wheat
../ml-research-loop/.venv/bin/ml-loop run --project . --config loop.efficientdet.toml \
  --state .loop-wheat --budget 4h --trials 2
```

学習はprivateのKaggleカーネルで行う。TorchVisionの公開事前学習済み重みを取得するため、
既定では学習カーネルのネット接続を許可する。データ本体はコンペ入力としてKaggleに接続し、
認証情報は学習コードに渡さない。検証の予測はローカルの固定評価コードで採点する。
小麦では `agent.aggregate_only = true` を設定している。Codexに渡す評価情報はスコアと
集計値に限り、誤検出画像、画像名、予測ファイル、実行時エラーの原文はローカルに残す。
`stop`は新規実験を止める。Kaggle上で既に開始したジョブは時間上限まで回収する。

### ローカルMLflowで学習状況を見る

[MLflowの実験一覧](http://localhost:5050)で `global-wheat-detection` を開く。
コンペごとに1つのExperimentを使う。学習は `train/<条件>/<試行ID>`、提出推論は
`infer/<条件>/<提出ID>`。RunのDescriptionにKaggle Notebookへのリンク、バージョン、実験条件を記録する。
Kaggleの新規Notebookも `wheat-train-<条件>-<試行IDのハッシュ>` と `wheat-infer-<条件>` で分ける。
一覧の `loop_status` は準備・実行・評価・終了の状態を示す。Runの `train_loss` はepochごとの学習loss、
`valid_loss` は固定の検証データで測ったloss、`optimizer_steps` は累積更新回数。
`wheat_image_precision` はローカルの固定検証スコア。同じ `comparison_group` のRunを比較する。
提出推論の `kaggle_public_score` と `kaggle_private_score` はKaggleの採点結果。
`--watch` は15秒ごとに同期し、実行中のlossはKaggleのログ配信後に表示する。画面は必要に応じて再読み込みする。

別のMacで動かす場合は、プロジェクト直下のターミナルで追跡サーバーを起動する。

```bash
mkdir -p .loop-wheat/mlflow
MLFLOW_DISABLE_TELEMETRY=true uv run --no-project --python 3.12 --with 'mlflow==3.18.0' mlflow server \
  --host 127.0.0.1 --port 5050 --workers 1 --allowed-hosts '127.0.0.1:5050,localhost:5050' \
  --backend-store-uri "sqlite:///$PWD/.loop-wheat/mlflow/tracking.sqlite" \
  --artifacts-destination "$PWD/.loop-wheat/mlflow/artifacts"
```

別のターミナルで同期を開始する。`--watch` を省くと保存済みの試行だけを1回取り込む。

```bash
MLFLOW_DISABLE_TELEMETRY=true uv run --script scripts/track_mlflow.py --watch
```

完了済みRunのArtifactsには評価結果、ログ、実行コードのNotebookを保存する。
`validation-previews/epoch-*` には固定の検証画像4枚の正解枠と予測枠を保存する。
途中epochの予測を保存していない過去の実験には `validation-previews/final` を表示する。
提出推論は重みを生成した学習Runのchildになる。`git_commit` は実行ソースのコミットを示す。
分割・画像・予測・重みの保管先は従来どおり `.loop-wheat/`。同期はLoopの状態データを読み取るため、同じ試行IDで再開できる。

### 保存済みの検証予測をアンサンブルする

同じ固定検証で完了した試行の成果物を渡し、Weighted Boxes Fusionで枠を統合する。
モデルの重みは初期値では均等。最初に指定したconfidenceで独立評価し、
confidenceを変えた各モデル単体のスコアと統合結果を `comparison.json` に保存する。
`--track` を付けると、ローカルのMLflowに比較元の学習Runと検出画像を記録する。

```bash
MLFLOW_DISABLE_TELEMETRY=true uv run --script scripts/ensemble_predictions.py \
  .loop-wheat/8fc98f3e68e8441a/artifacts/8fc98f3e68e8441a-0001 \
  .loop-wheat/8fc98f3e68e8441a/artifacts/8fc98f3e68e8441a-0002 \
  --output .loop-wheat/ensemble/mbv3-hflip-color15 --track
```

### 最終確認

合格候補を選んだ後に限り、その試行のソースと重みを使ってローカルで
`train.py --weights PATH --partition holdout`を実行し、別の成果物ディレクトリで
`evaluate.py --partition holdout`を実行する。両コマンドに`ML_LOOP_PROJECT`と
`ML_LOOP_ARTIFACTS`を渡す。自走中にholdoutは採点しない。

### Kaggleへの節目提出

固定検証で最良を更新した構成を選び、凍結した設定で独立確認を行う。
現行の提出構成を両分割で上回り、撮影元別の低下が0.01以内の候補で、学習とは別の非公開推論Notebookを作る。
推論はインターネット無効で、学習Notebook出力の重みと依存wheelをSHA-256で照合する。
検証用の10枚でCSV生成を確認後、Notebookの版を指定して提出する。
同じNotebook版を提出済みか履歴で照合する。Public・Privateは提出の測定値として記録し、モデル選択には固定検証を使う。
提出IDごとの記録は `.loop-wheat/inference/<提出ID>.json` に保存する。
`uv run --locked python scripts/refresh_submission_scores.py` で採点履歴を取り込み、MLflow同期が更新を反映する。

```bash
uv run --locked python scripts/build_submission_notebook.py \
  configs/submission-d2-mbv3-wbf040.json --output outputs/submission-d2-mbv3-wbf040
kaggle kernels push -p outputs/submission-d2-mbv3-wbf040 --accelerator NvidiaTeslaT4
# Notebookの実行成功と提出履歴を確認してから、実際に確認した版を指定する。
kaggle competitions submit -c global-wheat-detection \
  -k yutarotaro/wheat-infer-d2-mbv3-wbf040 -v 1 -f submission.csv -m 'd2-mbv3-wbf040 | val 0.580485 | v1'
```

## 公式公開版でのローカル検証

Kaggleの競技データを取得できない場合は、[GWHD 2020 v4](https://zenodo.org/records/4298502)の
公開済み学習画像と注釈を使ってローカルで検証できる。これは注釈が修正された別版で、
Kaggleのスコアとは比較しない。`data/gwhd2020-v4/`に分離し、未公開のテスト注釈は使わない。
小規模検証では撮影元を跨がない分割を固定し、使った画像枚数と設定を評価結果に記録する。

全データの実験には `loop.gwhd2020.toml` を使う。公式アーカイブを
`scripts/prepare_gwhd2020.py` で変換し、撮影元別の分割を固定する。Kaggleの非公開データセット
には学習・検証画像と学習用注釈を入れる。検証の正解注釈と最終確認用データはローカルに残す。
学習を始める前にコードと設定をコミットしてGitHubへpushし、作業ツリーをcleanにする。
同一コミットで複数設定を実行する場合は、それぞれのRun IDと設定を記録する。
Loopが生成した試行コミットは、ローカルの `refs/ml-loop/trials/` から参照できる。
各試行の仮説、比較元、採用条件、結果、次の判断は [実験記録](docs/experiments.md) から追えるようにする。

## Gitの運用

- 作業ブランチは`codex/`で始める。
- コミット・pushは、必要な検証を終えた作業の区切りで自分で判断して行う。毎回の確認は不要。
- デフォルトブランチへの直接コミット、rebase、force pushは行わない。
- push先とブランチを明示する。例: `git push -u origin codex/project-setup`。

設計: [プロジェクト構成](docs/specs/2026-11-14-project-setup-design.md)

実験記録: [experiments.md](docs/experiments.md)
