# PyCon JP Image Search

PyCon JP の Flickr アルバムから画像をダウンロードし、SigLIP 2 / CLIP-L による Embedding を生成して、テキストや画像による類似検索ができるシステムです。

サーバーサイドの Gradio UI に加え、Firebase Hosting でデプロイされる React Web アプリも提供しています。

## 機能概要

| 機能 | コマンド / ディレクトリ | 説明 |
|------|----------------------|------|
| 画像ダウンロード | `pyconjp-manage` | Flickr API でアルバム単位の画像取得 |
| Embedding 生成 | `pyconjp-embed generate` | SigLIP 2 / CLIP-L モデルで Embedding 生成 |
| 顔検出 | `pyconjp-embed face-generate` | InsightFace による顔検出・顔 Embedding 生成 |
| 物体検出 | `pyconjp-embed object-generate` | YOLO11 による物体検出・タグ付け |
| 検索 UI (Gradio) | `pyconjp-search` | Gradio ベースの検索 UI（サーバーサイド） |
| 検索 UI (React) | [`web/`](web/) | React + Vite の Web アプリ（Firebase Hosting） |

## Embedding モデル

| モデル | 次元 | DB ファイル | 用途 |
|--------|------|------------|------|
| SigLIP 2 Base | 768 | `pyconjp_image_search.duckdb` | Gradio UI のデフォルト |
| SigLIP 2 Large | 1024 | `pyconjp_image_search_siglip2_large.duckdb` | 高精度検索 |
| CLIP-L | 768 | `pyconjp_image_search_clip.duckdb` | React Web アプリ（ブラウザ内実行） |
| InsightFace | 512 | (顔検出用) | 顔類似検索 |
| YOLO11s | - | (物体検出用) | COCO 80 クラスのタグ付け |

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
uv sync              # 基本パッケージのみ
uv sync --extra ml   # ML 系パッケージ（Embedding 生成・物体検出に必要）
```

### 環境変数

```bash
cp .env.example .env
```

`.env` に Flickr API キーとユーザー ID を設定してください。

## クイックスタート

```bash
# 1. DB 初期化
uv run pyconjp-manage init-db

# 2. 画像ダウンロード
uv run pyconjp-manage download-flickr \
    --album-id <ALBUM_ID> --event "PyCon JP 2024" --year 2024

# 3. Embedding 生成
uv run pyconjp-embed generate --batch-size 32

# 4. 検索 UI 起動
uv run pyconjp-search
```

## プロジェクト構成

```
pyconjp-image-search/
├── src/pyconjp_image_search/   # Python パッケージ
│   ├── manager/                # 画像管理 (Flickr ダウンロード)
│   ├── embedding/              # Embedding・顔検出・物体検出
│   └── search/                 # Gradio 検索 UI
├── scripts/                    # ユーティリティスクリプト
├── web/                        # React Web アプリ
├── docs/                       # 詳細ドキュメント
└── data/                       # データファイル (gitignore)
```

## 技術スタック

### バックエンド (Python)

| 用途 | ライブラリ |
|------|-----------|
| パッケージ管理 | uv + hatchling |
| DB | DuckDB |
| Flickr API | httpx |
| Embedding | SigLIP 2 / CLIP-L (transformers + torch) |
| 顔検出 | InsightFace (buffalo_l) |
| 物体検出 | YOLO11s (ultralytics) |
| 類似検索 | DuckDB `list_cosine_similarity` |
| 検索 UI | Gradio |

### フロントエンド (React Web アプリ)

| 用途 | ライブラリ |
|------|-----------|
| フレームワーク | React 19 + TypeScript |
| ビルド | Vite 6 |
| Embedding | Transformers.js (`@huggingface/transformers`) |
| DB | DuckDB WASM / Firebase Firestore |
| 認証 | Firebase Auth |
| ホスティング | Firebase Hosting |

## ドキュメント

| ドキュメント | 内容 |
|-------------|------|
| [docs/cli-usage.md](docs/cli-usage.md) | CLI コマンドリファレンス |
| [docs/database-schema.md](docs/database-schema.md) | DuckDB テーブル定義 |
| [docs/voronoi-search.md](docs/voronoi-search.md) | 顔検索の高速化（Voronoi 分割） |
| [web/README.md](web/README.md) | React Web アプリ |
| [web/FIREBASE.md](web/FIREBASE.md) | Firebase / Firestore 設定 |
