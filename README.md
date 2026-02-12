# PyCon JP Image Search

PyCon JP の Flickr アルバムから画像をダウンロードし、SigLIP による Embedding を生成して、テキストや画像による類似検索ができるシステムです。

## 機能概要

| 機能 | CLI コマンド | 説明 |
|------|-------------|------|
| 画像ダウンロード | `pyconjp-manage` | Flickr API でアルバム単位の画像取得、DuckDB にメタデータ保存 |
| Embedding 生成 | `pyconjp-embed` | SigLIP モデルで画像の Embedding ベクトルを生成 |
| 検索 UI | `pyconjp-search` | Gradio ベースの検索インターフェース |

## セットアップ

### 必要環境

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- CUDA 対応 GPU（Embedding 生成時）

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
uv run pyconjp-embed status
```

#### Embedding の生成

```bash
uv run pyconjp-embed generate --batch-size 32
```

SigLIP (`google/siglip-base-patch16-224`) を使って 768 次元の Embedding ベクトルを生成し、DuckDB に保存します。未処理の画像のみ処理されるため、中断後の再実行も安全です。

**オプション:**

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--batch-size` | `32` | バッチサイズ |
| `--device` | `cuda` | デバイス (`cuda` / `cpu`) |
| `--model` | `google/siglip-base-patch16-224` | モデル名 |

### 4. 検索 UI

```bash
uv run pyconjp-search
```

Gradio ベースの Web UI が起動します（デフォルト: http://localhost:7860）。

#### Text Search タブ

テキストで画像を検索します（例: "keynote speaker on stage"）。SigLIP モデルでテキストを Embedding に変換し、コサイン類似度で検索します。

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

- **Search Cropped** -- 選択した矩形領域をクロップしてサーバー側で SigLIP の Embedding を生成し、その領域に類似する画像を Image Search タブで検索
- **Copy to Clipboard** -- 選択した矩形領域をクリップボードにコピー

矩形が選択されるまで Search Cropped・Copy to Clipboard ボタンは無効化されます。

#### 画像表示

検索結果の画像は Flickr の静的 CDN URL から直接表示されます（ローカルファイル不要）。ギャラリーには 640px サイズ、プレビューには 1024px サイズが使用されます。

モデルは初回検索時に自動ロードされます。

## プロジェクト構成

```
pyconjp-image-search/
├── pyproject.toml
├── .env.example
├── .gitignore
├── pyconjp_image_search.duckdb      # DuckDB データベース (gitignore)
├── data/pyconjp/                     # ダウンロード画像 (gitignore)
│   ├── pycon_jp_2024_conference_day1/
│   │   ├── 53912345678.jpg
│   │   └── ...
│   └── ...
├── scripts/
│   └── download_all.py               # 全アルバム一括ダウンロード
└── src/pyconjp_image_search/
    ├── __init__.py
    ├── config.py                      # 設定（DB パス、Flickr API、Embedding モデル）
    ├── db.py                          # DuckDB 接続ファクトリ
    ├── models.py                      # ImageMetadata dataclass
    ├── manager/                       # 画像管理モジュール
    │   ├── __init__.py                # CLI エントリポイント (pyconjp-manage)
    │   ├── schema.py                  # DDL・マイグレーション
    │   ├── flickr_client.py           # Flickr REST API クライアント
    │   ├── downloader.py              # アルバムダウンローダー
    │   └── repository.py              # images テーブル CRUD
    ├── embedding/                     # Embedding モジュール
    │   ├── __init__.py                # CLI エントリポイント (pyconjp-embed)
    │   ├── siglip.py                  # SigLIPEmbedder クラス
    │   └── repository.py              # image_embeddings テーブル CRUD
    └── search/                        # 検索 UI モジュール
        ├── __init__.py                # CLI エントリポイント (pyconjp-search)
        ├── query.py                   # 検索クエリ
        └── app.py                     # Gradio アプリ
```

## データベース構成

DuckDB を使用し、プロジェクトルートに単一ファイル (`pyconjp_image_search.duckdb`) として保存されます。

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

SigLIP で生成した Embedding ベクトルを保存します。

| カラム | 型 | 説明 |
|-------|-----|------|
| `image_id` | INTEGER (PK, FK) | images.id への外部キー |
| `model_name` | VARCHAR (PK) | モデル名 |
| `embedding` | FLOAT[768] | 768 次元の Embedding ベクトル (L2 正規化済み) |
| `created_at` | TIMESTAMP | レコード作成日時 |

複合主キー `(image_id, model_name)` により、将来的に複数モデルの Embedding を同時に保持できます。

## 技術スタック

| 用途 | ライブラリ |
|------|-----------|
| パッケージ管理 | uv + hatchling |
| DB | DuckDB |
| Flickr API | httpx |
| Embedding | SigLIP (transformers + torch) |
| 類似検索 | DuckDB `list_cosine_similarity` |
| 検索 UI | Gradio |
| 進捗表示 | rich |
| リトライ | tenacity |
| 画像処理 | Pillow |
