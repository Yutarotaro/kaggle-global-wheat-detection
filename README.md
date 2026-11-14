# Global Wheat Detection

小麦の穂をバウンディングボックスで検出する、Kaggleの過去コンペに取り組む。

- コンペ: https://www.kaggle.com/competitions/global-wheat-detection
- コード・設定・実験記録の本体はこのローカルリポジトリ。
- GitHubのprivateリポジトリでコードと記録を管理する。
- データ確認と可視化はMac、本格的な学習はKaggle GPUで行う。

## 現在の段階

[ML Research Loop](https://github.com/Yutarotaro/ml-research-loop)に接続済み。
固定分割、採点、GPU向け学習・推論、独立評価によるエラー画像出力を実装した。
合成画像でCPU学習から採点までを検証済み。実コンペの注釈CSVの取得は403で、実データの学習と4時間の実験は未実施。

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

学習はprivateのKaggleカーネルで行う。TorchVisionの公開事前学習済み重みを取得するため、
既定では学習カーネルのネット接続を許可する。データ本体はコンペ入力としてKaggleに接続し、
認証情報は学習コードに渡さない。検証の予測はローカルの固定評価コードで採点する。
`stop`は新規実験を止める。Kaggle上で既に開始したジョブは時間上限まで回収する。

合格候補を選んだ後に限り、その試行のソースと重みを使ってローカルで
`train.py --weights PATH --partition holdout`を実行し、別の成果物ディレクトリで
`evaluate.py --partition holdout`を実行する。両コマンドに`ML_LOOP_PROJECT`と
`ML_LOOP_ARTIFACTS`を渡す。自走中にholdoutは採点しない。

## Gitの運用

- 作業ブランチは`codex/`で始める。
- コミット・pushは、必要な検証を終えた作業の区切りで自分で判断して行う。毎回の確認は不要。
- デフォルトブランチへの直接コミット、rebase、force pushは行わない。
- push先とブランチを明示する。例: `git push -u origin codex/project-setup`。

設計: [プロジェクト構成](docs/specs/2026-11-14-project-setup-design.md)

実験記録: [experiments.md](docs/experiments.md)
