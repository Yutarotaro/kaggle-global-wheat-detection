# Global Wheat Detection

小麦の穂をバウンディングボックスで検出する、Kaggleの過去コンペに取り組む。

- コンペ: https://www.kaggle.com/competitions/global-wheat-detection
- コード・設定・実験記録の本体はこのローカルリポジトリ。
- GitHubのprivateリポジトリでコードと記録を管理する。
- データ確認と可視化はMac、本格的な学習はKaggle GPUで行う。

## 現在の段階

プロジェクト基盤の準備段階。データ取得、可視化スクリプト、検証分割、評価実装、モデル学習はまだ行っていない。
PyTorchと検出ライブラリは、データ確認とモデル選定後に追加する。

2026-11-14に確認済み:

- Python 3.12環境の構築、初期依存のimport、画像の描画と読み込み。
- Kaggle CLI 2.2.5の起動と、設定済み認証を使った小麦コンペのファイル一覧取得。
- GitHubリポジトリのprivate設定とローカルのorigin接続。

データ本体のダウンロード可否とKaggle GPUの利用枠はまだ確認していない。

## 環境

Python 3.12を使用する。依存パッケージの解決結果は`uv.lock`に保存する。

```bash
uv sync --locked --group kaggle
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

## Kaggleでの学習

API経由でローカルのスクリプトまたはNotebookと`kernel-metadata.json`を送信する。
Kaggleの実行設定はprivateとし、コンペの入力は`competition_sources`から参照する。
現在はまだ実行ファイルを作っていない。認証・GPU利用枠の確認後に準備する。

モデルや結果を`/kaggle/working/`に保存し、実行完了後にローカルへ取得する。
ローカルの`uv.lock`とKaggleの事前構築環境は同一ではないため、実行時のライブラリバージョンも記録する。

## Gitの運用

- 作業ブランチは`codex/`で始める。
- コミット・pushは、必要な検証を終えた作業の区切りで自分で判断して行う。毎回の確認は不要。
- デフォルトブランチへの直接コミット、rebase、force pushは行わない。
- push先とブランチを明示する。例: `git push -u origin codex/project-setup`。

設計: [プロジェクト構成](docs/specs/2026-11-14-project-setup-design.md)

実験記録: [experiments.md](docs/experiments.md)
