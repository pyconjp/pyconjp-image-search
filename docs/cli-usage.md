# CLI コマンドリファレンス

## 画像管理 (`pyconjp-manage`)

### データベース初期化

```bash
uv run pyconjp-manage init-db
```

プロジェクトルートに `pyconjp_image_search.duckdb` が作成されます。

### アルバム一覧の確認

```bash
uv run pyconjp-manage list-albums
```

### 画像のダウンロード

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

画像は `data/pyconjp/<album_title>/` にアルバムごとに保存されます。ファイル名は Flickr の photo ID (`<photo_id>.jpg`) です。2回目以降は増分ダウンロード（既存画像はスキップ）されます。

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

### DB 内の画像一覧

```bash
uv run pyconjp-manage list --event "PyCon JP 2024"
uv run pyconjp-manage list --year 2024
uv run pyconjp-manage list --album-id 72177720322202729
```

## Embedding 生成 (`pyconjp-embed`)

### 生成状況の確認

```bash
# SigLIP 2 (デフォルト)
uv run pyconjp-embed status

# CLIP-L
uv run pyconjp-embed status --model clip
```

### Embedding の生成

```bash
# SigLIP 2 (デフォルト)
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
| `--model` | `siglip` | モデル選択 (`siglip` / `siglip-large` / `clip`) |
| `--limit` | 全件 | 処理する最大画像数 |
| `--force` | - | 既存 Embedding を上書き再生成 |

## 顔検出 (`pyconjp-embed face-generate`)

InsightFace による顔検出・顔 Embedding 生成（512 次元）。

```bash
uv run pyconjp-embed face-generate
uv run pyconjp-embed face-status
```

## 物体検出 (`pyconjp-embed object-generate`)

YOLO11 により画像内の物体を検出し、COCO 80 クラスのラベル（person, chair, laptop 等）をタグとして DB に保存します。

```bash
uv run pyconjp-embed object-generate
uv run pyconjp-embed object-status
```

**オプション:**

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--limit` | 全件 | 処理する最大画像数 |
| `--force` | - | 既存の検出結果を削除して再実行 |
| `--commit-interval` | `50` | N 画像ごとに DB にコミット |

## 検索 UI (`pyconjp-search`)

```bash
uv run pyconjp-search
```

Gradio ベースの Web UI が起動します（デフォルト: http://localhost:7860）。

### Text Search タブ

テキストで画像を検索します（例: "keynote speaker on stage"）。選択中のモデルでテキストを Embedding に変換し、コサイン類似度で検索します。

- **モデル切り替え** -- ドロップダウンで SigLIP 2 / CLIP-L を選択
- **イベントフィルター** -- ドロップダウンでイベント名を選択して絞り込み
- **プレビュー** -- 検索結果の画像をクリックすると拡大プレビュー表示
- **サムネイルストリップ** -- プレビュー下部に検索結果のサムネイル一覧を表示
- **Load More** -- ページネーション（20件ずつ追加読み込み）

### Image Search タブ

画像をアップロードして類似画像を検索します。機能は Text Search と同様です。

### Find Similar（類似画像検索）

プレビュー表示中に **Find Similar** ボタンをクリックすると、選択中の画像の DB に保存済みの Embedding を使って類似検索を実行します。

### クロップ機能

プレビュー画像上でマウスドラッグにより矩形を選択できます。

- **Search Cropped** -- 選択した矩形領域をクロップしてサーバー側で Embedding を生成し、類似画像を検索
- **Copy to Clipboard** -- 選択した矩形領域をクリップボードにコピー

### 画像表示

検索結果の画像は Flickr の静的 CDN URL から直接表示されます（ローカルファイル不要）。ギャラリーには 640px サイズ、プレビューには 1024px サイズが使用されます。モデルは初回検索時に自動ロードされます。

## スクリプト

### 全アルバム一括ダウンロード

```bash
uv run scripts/download_all.py
```

### Voronoi パーティション割り当て

```bash
uv run scripts/assign_voronoi_partitions.py [--force]
```

詳細は [Voronoi 検索](voronoi-search.md) を参照。

### Firestore へのデータ投入

```bash
uv run scripts/upload_to_firestore.py [--dry-run]
```

詳細は [web/FIREBASE.md](../web/FIREBASE.md) を参照。
