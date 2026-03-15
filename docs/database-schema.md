# データベーススキーマ

DuckDB を使用し、モデルごとに別ファイルとして保存されます。

| DB ファイル | モデル | 用途 |
|------------|--------|------|
| `pyconjp_image_search.duckdb` | SigLIP 2 Base | Gradio UI のデフォルト |
| `pyconjp_image_search_siglip2_large.duckdb` | SigLIP 2 Large | 高精度検索 |
| `pyconjp_image_search_clip.duckdb` | CLIP-L | React Web アプリ（ブラウザ内実行） |

全 DB が同じスキーマを持ちます。画像メタデータ・顔検出・物体検出は全 DB で共通、Embedding のみモデル固有です。

## images テーブル

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

## image_embeddings テーブル

Embedding ベクトルを保存します。

| カラム | 型 | 説明 |
|-------|-----|------|
| `image_id` | INTEGER (PK, FK) | images.id への外部キー |
| `model_name` | VARCHAR (PK) | モデル名 |
| `embedding` | FLOAT[768] | 768 次元の Embedding ベクトル (L2 正規化済み) |
| `created_at` | TIMESTAMP | レコード作成日時 |

複合主キー `(image_id, model_name)` により、複数モデルの Embedding を同時に保持できます。

## face_detections テーブル

InsightFace による顔検出結果と顔 Embedding を保存します。

| カラム | 型 | 説明 |
|-------|-----|------|
| `face_id` | VARCHAR (PK) | 顔の一意 ID |
| `image_id` | INTEGER (FK) | images.id への外部キー |
| `model_name` | VARCHAR | モデル名 (`insightface/buffalo_l`) |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | FLOAT | バウンディングボックス |
| `det_score` | FLOAT | 検出スコア |
| `landmark` | VARCHAR | 顔ランドマーク |
| `age` | INTEGER | 推定年齢 (nullable) |
| `gender` | VARCHAR | 性別 "M"/"F" (nullable) |
| `embedding` | FLOAT[512] | 顔 Embedding (ArcFace 512 次元) |
| `person_label` | VARCHAR | 人物ラベル (nullable) |
| `cluster_id` | INTEGER | クラスタ ID (nullable) |
| `voronoi_partition_ids` | INTEGER[] | Voronoi パーティション ID 配列 |

## object_detections テーブル

YOLO11 による物体検出結果を保存します。

| カラム | 型 | 説明 |
|-------|-----|------|
| `detection_id` | VARCHAR (PK) | 検出 ID（`{image_id}_{model}_{index}` 形式） |
| `image_id` | INTEGER (FK) | images.id への外部キー |
| `model_name` | VARCHAR | モデル名（`yolo11s`） |
| `label` | VARCHAR | COCO 80 クラスのラベル（person, chair 等） |
| `confidence` | FLOAT | 検出信頼度 |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | FLOAT | バウンディングボックス |
| `created_at` | TIMESTAMP | レコード作成日時 |

## object_processed_images テーブル

物体検出の処理済み画像を管理します。

| カラム | 型 | 説明 |
|-------|-----|------|
| `image_id` | INTEGER (PK, FK) | images.id への外部キー |
| `model_name` | VARCHAR (PK) | モデル名 |
| `object_count` | INTEGER | 検出された物体数 |
| `processed_at` | TIMESTAMP | 処理日時 |

## Firestore コレクション

Firestore へのマッピングについては [web/FIREBASE.md](../web/FIREBASE.md) を参照してください。
