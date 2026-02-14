# PyCon JP Image Search

PyCon JP の Flickr アルバムから画像をダウンロードし、SigLIP / CLIP-L による Embedding を生成して、テキストや画像による類似検索ができるシステムです。

サーバーサイドの Gradio UI に加え、ブラウザ内で完結する React Web アプリも提供しています。

## 機能概要

| 機能 | CLI コマンド / ディレクトリ | 説明 |
|------|---------------------------|------|
| 画像ダウンロード | `pyconjp-manage` | Flickr API でアルバム単位の画像取得、DuckDB にメタデータ保存 |
| Embedding 生成 | `pyconjp-embed` | SigLIP / CLIP-L モデルで画像の Embedding ベクトルを生成 |
| 検索 UI (Gradio) | `pyconjp-search` | Gradio ベースの検索インターフェース（サーバーサイド） |
| 検索 UI (React) | `web/` | React + Vite のクライアントサイド検索アプリ |

## Embedding モデル

2 つのモデルをサポートしています。どちらも 768 次元の Embedding ベクトルを生成します。

| モデル | モデル名 | DB ファイル | 用途 |
|--------|---------|------------|------|
| SigLIP | `google/siglip-base-patch16-224` | `pyconjp_image_search.duckdb` | Gradio UI のデフォルト |
| CLIP-L | `openai/clip-vit-large-patch14` | `pyconjp_image_search_clip.duckdb` | React Web アプリ、Gradio UI で選択可能 |

Gradio UI ではドロップダウンで SigLIP / CLIP-L を切り替えて検索できます。React Web アプリは CLIP-L のみ使用します（ブラウザ内で Transformers.js によりモデルを実行）。

## セットアップ

### 必要環境

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- CUDA 対応 GPU（Embedding 生成時）
- Node.js 24+（React Web アプリのビルド時）

### インストール

```bash
git clone <repository-url>
cd pyconjp-image-search
uv sync
```

### 環境変数の設定

```bash
cp .env.example .env
```

`.env` に以下を設定:

```
FLICKR_API_KEY=your_api_key_here
FLICKR_USER_ID=your_flickr_user_id
```

Flickr API キーは https://www.flickr.com/services/api/keys/ から取得できます。

## 使い方

### 1. データベース初期化

```bash
uv run pyconjp-manage init-db
```

プロジェクトルートに `pyconjp_image_search.duckdb` が作成されます。

### 2. 画像のダウンロード

#### アルバム一覧の確認

```bash
uv run pyconjp-manage list-albums
```

#### アルバムのダウンロード

```bash
# dry-run で枚数確認
uv run pyconjp-manage download-flickr \
    --album-id 72177720322202729 \
    --event "PyCon JP 2024" --year 2024 --dry-run

# 実際にダウンロード
uv run pyconjp-manage download-flickr \
    --album-id 72177720322202729 \
    --event "PyCon JP 2024" --year 2024
```

画像は `data/pyconjp/<album_title>/` にアルバムごとに保存されます。ファイル名は Flickr の photo ID (`<photo_id>.jpg`) です。

**オプション:**

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--user-id` | `.env` の `FLICKR_USER_ID` | Flickr ユーザー ID |
| `--album-id` | (必須) | Flickr アルバム ID |
| `--album-title` | 自動検出 | アルバムタイトル（ディレクトリ名に使用） |
| `--event` | (必須) | イベント名 |
| `--year` | (必須) | イベント年 |
| `--event-type` | `conference` | イベント種別 |
| `--size` | `b` (1024px) | 画像サイズ (`s,q,t,m,z,b,h,k,o`) |
| `--dry-run` | - | ダウンロードせず枚数のみ表示 |

2回目以降は増分ダウンロード（既存画像はスキップ）されます。

#### DB 内の画像一覧

```bash
uv run pyconjp-manage list --event "PyCon JP 2024"
uv run pyconjp-manage list --year 2024
uv run pyconjp-manage list --album-id 72177720322202729
```

### 3. Embedding 生成

#### 生成状況の確認

```bash
# SigLIP (デフォルト)
uv run pyconjp-embed status

# CLIP-L
uv run pyconjp-embed status --model clip
```

#### Embedding の生成

```bash
# SigLIP (デフォルト)
uv run pyconjp-embed generate --batch-size 32

# CLIP-L
uv run pyconjp-embed generate --model clip --batch-size 32
```

指定モデルで 768 次元の Embedding ベクトルを生成し、対応する DuckDB に保存します。未処理の画像のみ処理されるため、中断後の再実行も安全です。

**オプション:**

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--batch-size` | `32` | バッチサイズ |
| `--device` | `cuda` | デバイス (`cuda` / `cpu`) |
| `--model` | `siglip` | モデル選択 (`siglip` / `clip`) |
| `--limit` | 全件 | 処理する最大画像数 |
| `--force` | - | 既存 Embedding を上書き再生成 |

### 4. 検索 UI (Gradio)

```bash
uv run pyconjp-search
```

Gradio ベースの Web UI が起動します（デフォルト: http://localhost:7860）。

#### Text Search タブ

テキストで画像を検索します（例: "keynote speaker on stage"）。選択中のモデルでテキストを Embedding に変換し、コサイン類似度で検索します。

- **モデル切り替え** -- ドロップダウンで SigLIP / CLIP-L を選択
- **イベントフィルター** -- ドロップダウンでイベント名を選択して絞り込み
- **プレビュー** -- 検索結果の画像をクリックすると拡大プレビュー表示
- **サムネイルストリップ** -- プレビュー下部に検索結果のサムネイル一覧を表示
- **Load More** -- ページネーション（20件ずつ追加読み込み）

#### Image Search タブ

画像をアップロードして類似画像を検索します。機能は Text Search と同様です。

#### Find Similar（類似画像検索）

Text Search・Image Search どちらのタブでも、プレビュー表示中に **Find Similar** ボタンをクリックすると：

- 選択中の画像の DB に保存済みの Embedding を使って類似検索を実行
- Image Search タブに自動切り替え
- イベントフィルターの条件を引き継ぎ
- 検索元の画像を Image Search のアップロード欄に表示

#### クロップ機能

プレビュー画像上でマウスドラッグにより矩形を選択できます。

- **Search Cropped** -- 選択した矩形領域をクロップしてサーバー側で Embedding を生成し、その領域に類似する画像を Image Search タブで検索
- **Copy to Clipboard** -- 選択した矩形領域をクリップボードにコピー

矩形が選択されるまで Search Cropped・Copy to Clipboard ボタンは無効化されます。

#### 画像表示

検索結果の画像は Flickr の静的 CDN URL から直接表示されます（ローカルファイル不要）。ギャラリーには 640px サイズ、プレビューには 1024px サイズが使用されます。

モデルは初回検索時に自動ロードされます。

### 5. 検索 UI (React Web アプリ)

サーバー不要で、すべての処理がブラウザ内で完結するクライアントサイドアプリケーションです。

#### セットアップ

```bash
cd web
npm install
npm run dev
```

開発サーバーが起動します（デフォルト: http://localhost:5173）。

本番ビルド:

```bash
npm run build
npm run preview
```

#### 必要なデータ

React アプリは CLIP-L の DuckDB ファイル (`pyconjp_image_search_clip.duckdb`) を公開 URL から取得してブラウザ内の DuckDB WASM で読み込みます。事前に CLIP-L の Embedding 生成が必要です。

#### アーキテクチャ

- **CLIP モデル**: `Xenova/clip-vit-large-patch14` を Transformers.js (`@huggingface/transformers`) でブラウザ内実行
- **データベース**: DuckDB WASM (`@duckdb/duckdb-wasm`) でブラウザ内クエリ
- **ベクトル検索**: `list_cosine_similarity` によるコサイン類似度検索
- **画像表示**: Flickr 静的 CDN URL から直接表示

#### 機能

- **テキスト検索** -- 検索テキストをブラウザ内で CLIP Embedding に変換し、類似画像を検索
- **英訳ボタン** -- Chrome Translator API (Chrome 138+) による日本語→英語翻訳。CLIP-L は英語に最適化されているため、日本語テキストを英訳してから検索すると精度が向上。非対応ブラウザではボタンが非活性になり、ツールチップで案内を表示
- **画像検索** -- 画像アップロードまたはクリップボードからの貼り付けで類似画像を検索（Vision モデルは初回利用時に遅延ロード）
- **イベントフィルター** -- イベント名で絞り込み
- **プレビュー** -- 画像クリックで拡大表示、サムネイルストリップ付き
- **Find Similar** -- 検索結果の画像から類似画像を再検索
- **クロップ検索** -- プレビュー画像上でドラッグして矩形選択し、その領域で類似検索
- **Load More** -- ページネーション

## プロジェクト構成

```
pyconjp-image-search/
├── pyproject.toml
├── .env.example
├── .gitignore
├── pyconjp_image_search.duckdb          # SigLIP 用 DB (gitignore)
├── pyconjp_image_search_clip.duckdb     # CLIP-L 用 DB (gitignore)
├── data/pyconjp/                        # ダウンロード画像 (gitignore)
│   ├── pycon_jp_2024_conference_day1/
│   │   ├── 53912345678.jpg
│   │   └── ...
│   └── ...
├── scripts/
│   ├── download_all.py                  # 全アルバム一括ダウンロード
│   └── copy_metadata_to_clip_db.py      # SigLIP DB → CLIP DB へメタデータコピー
├── web/                                 # React Web アプリ
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── App.tsx                      # メインコンポーネント
│       ├── App.css                      # スタイル
│       ├── components/                  # UI コンポーネント
│       │   ├── SearchBar.tsx            # テキスト検索バー (英訳ボタン付き)
│       │   ├── ImageUpload.tsx          # 画像アップロード
│       │   ├── Gallery.tsx              # 検索結果グリッド
│       │   ├── Preview.tsx              # プレビュー表示
│       │   ├── CropOverlay.tsx          # クロップ選択 UI
│       │   ├── EventFilter.tsx          # イベントフィルター
│       │   └── ...
│       ├── hooks/                       # カスタムフック
│       │   ├── useImageSearch.ts        # 検索ロジック
│       │   ├── useCLIPEncoder.ts        # CLIP モデル管理
│       │   └── useDuckDB.ts            # DuckDB 初期化
│       ├── lib/                         # ユーティリティ
│       │   ├── search.ts               # DuckDB クエリ・ベクトル検索
│       │   ├── clip.ts                 # CLIP モデルラッパー
│       │   ├── duckdb.ts               # DuckDB WASM セットアップ
│       │   └── flickr.ts               # Flickr URL リサイズ
│       └── types/
│           ├── index.ts                 # TypeScript 型定義
│           └── translator.d.ts          # Chrome Translator API 型定義
└── src/pyconjp_image_search/
    ├── __init__.py
    ├── config.py                        # 設定（DB パス、Flickr API、Embedding モデル）
    ├── db.py                            # DuckDB 接続ファクトリ
    ├── models.py                        # ImageMetadata dataclass
    ├── manager/                         # 画像管理モジュール
    │   ├── __init__.py                  # CLI エントリポイント (pyconjp-manage)
    │   ├── schema.py                    # DDL・マイグレーション
    │   ├── flickr_client.py             # Flickr REST API クライアント
    │   ├── downloader.py                # アルバムダウンローダー
    │   └── repository.py                # images テーブル CRUD
    ├── embedding/                       # Embedding モジュール
    │   ├── __init__.py                  # CLI エントリポイント (pyconjp-embed)
    │   ├── siglip.py                    # SigLIPEmbedder クラス
    │   ├── clip.py                      # CLIPEmbedder クラス
    │   └── repository.py                # image_embeddings テーブル CRUD
    └── search/                          # 検索 UI モジュール
        ├── __init__.py                  # CLI エントリポイント (pyconjp-search)
        ├── query.py                     # 検索クエリ
        └── app.py                       # Gradio アプリ
```

## データベース構成

DuckDB を使用し、モデルごとに別ファイルとして保存されます。

- **SigLIP 用**: `pyconjp_image_search.duckdb`
- **CLIP-L 用**: `pyconjp_image_search_clip.duckdb`

どちらも同じスキーマを持ちます。`scripts/copy_metadata_to_clip_db.py` で SigLIP 用 DB から CLIP-L 用 DB へメタデータをコピーできます。

### images テーブル

画像のメタデータを管理します。

| カラム | 型 | 説明 |
|-------|-----|------|
| `id` | INTEGER (PK) | 自動採番 |
| `image_url` | VARCHAR (UNIQUE) | Flickr 画像 URL |
| `relative_path` | VARCHAR | `data/pyconjp/` からの相対パス |
| `local_filename` | VARCHAR | ファイル名 |
| `flickr_photo_id` | VARCHAR (UNIQUE) | Flickr photo ID |
| `album_id` | VARCHAR | Flickr アルバム ID |
| `album_title` | VARCHAR | アルバムタイトル |
| `event_name` | VARCHAR | イベント名 |
| `event_year` | INTEGER | イベント年 |
| `event_type` | VARCHAR | イベント種別 (default: `conference`) |
| `image_format` | VARCHAR | 画像フォーマット (JPEG 等) |
| `width` | INTEGER | 画像幅 (px) |
| `height` | INTEGER | 画像高さ (px) |
| `file_size_bytes` | BIGINT | ファイルサイズ |
| `downloaded_at` | TIMESTAMP | ダウンロード日時 |
| `created_at` | TIMESTAMP | レコード作成日時 |

### image_embeddings テーブル

Embedding ベクトルを保存します。

| カラム | 型 | 説明 |
|-------|-----|------|
| `image_id` | INTEGER (PK, FK) | images.id への外部キー |
| `model_name` | VARCHAR (PK) | モデル名 |
| `embedding` | FLOAT[768] | 768 次元の Embedding ベクトル (L2 正規化済み) |
| `created_at` | TIMESTAMP | レコード作成日時 |

複合主キー `(image_id, model_name)` により、複数モデルの Embedding を同時に保持できます。

## 技術スタック

### バックエンド (Python)

| 用途 | ライブラリ |
|------|-----------|
| パッケージ管理 | uv + hatchling |
| DB | DuckDB |
| Flickr API | httpx |
| Embedding | SigLIP / CLIP-L (transformers + torch) |
| 類似検索 | DuckDB `list_cosine_similarity` |
| 検索 UI | Gradio |
| 進捗表示 | rich |
| リトライ | tenacity |
| 画像処理 | Pillow |

### フロントエンド (React Web アプリ)

| 用途 | ライブラリ |
|------|-----------|
| フレームワーク | React 19 + TypeScript |
| ビルド | Vite 6 |
| Embedding | Transformers.js (`@huggingface/transformers`) |
| DB | DuckDB WASM (`@duckdb/duckdb-wasm`) |
| 翻訳 | Chrome Translator API (Chrome 138+) |
