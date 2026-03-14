# Firebase 設定

## 基本情報

| 項目 | 値 |
|------|-----|
| プロジェクトID | `pyconjp-image-search` |
| Hosting 公開ディレクトリ | `dist` (Vite ビルド出力) |
| Firestore データベース | `(default)` |
| Firestore リージョン | `asia-northeast1` |
| 認証プロバイダー | Google Sign-In |

### Firestore セキュリティルール

- **書き込み**: すべて禁止
- **読み込み**: `@pycon.jp` ドメインのメールアドレスを持つ、メール検証済みの認証ユーザーのみ許可

## デプロイ方法

### 自動デプロイ (GitHub Actions)

2つのワークフローが設定されています。

#### PR 時 (プレビューデプロイ)

- ファイル: `.github/workflows/firebase-hosting-pull-request.yml`
- トリガー: Pull Request の作成・更新時
- 動作: Firebase Hosting のプレビューチャンネルにデプロイし、PR コメントにプレビュー URL を投稿

#### マージ時 (本番デプロイ)

- ファイル: `.github/workflows/firebase-hosting-merge.yml`
- トリガー: `main` ブランチへの push (PR マージ)
- 動作: Firebase Hosting の本番 (`live`) チャンネルにデプロイ

### 手動デプロイ

```bash
cd web
npm ci && npm run build
npx firebase deploy
```

Hosting のみデプロイする場合:

```bash
npx firebase deploy --only hosting
```

Firestore ルールのみデプロイする場合:

```bash
npx firebase deploy --only firestore:rules
```

Firestore インデックスのみデプロイする場合:

```bash
npx firebase deploy --only firestore:indexes
```

## その他説明

### プロジェクトの確認

```bash
npx firebase projects:list
```

```bash
$ npx firebase projects:list
✔ Preparing the list of your Firebase projects
┌──────────────────────┬────────────────────────────────┬────────────────┬──────────────────────┐
│ Project Display Name │ Project ID                     │ Project Number │ Resource Location ID │
├──────────────────────┼────────────────────────────────┼────────────────┼──────────────────────┤
│ pyconjp-image-search │ pyconjp-image-search (current) │ 786650914822   │ [Not specified]      │
└──────────────────────┴────────────────────────────────┴────────────────┴──────────────────────┘
```

### ファイル構成

| ファイル | 説明 |
|----------|------|
| `firebase.json` | Firebase の設定ファイル (Hosting, Firestore, Auth) |
| `.firebaserc` | デフォルトプロジェクトの紐付け |
| `firestore.rules` | Firestore セキュリティルール |
| `firestore.indexes.json` | Firestore インデックス定義 |

## DuckDB スキーマと Firestore マッピング

### DuckDB 現行スキーマ

データは `pyconjp_image_search.duckdb` に格納されている。

#### `images` テーブル — 画像メタデータ

| カラム | 型 | 説明 |
|--------|-----|------|
| `id` | INTEGER PK | 自動採番 |
| `image_url` | VARCHAR NOT NULL | Flickr 画像 URL |
| `flickr_photo_id` | VARCHAR UNIQUE | Flickr 写真 ID |
| `album_id` | VARCHAR | Flickr アルバム ID |
| `album_title` | VARCHAR | アルバムタイトル |
| `event_name` | VARCHAR NOT NULL | イベント名 |
| `event_year` | INTEGER NOT NULL | イベント年 |
| `event_type` | VARCHAR | イベント種別 (default: 'conference') |
| `width` | INTEGER | 画像幅 (px) |
| `height` | INTEGER | 画像高さ (px) |

#### `image_embeddings` テーブル — ベクトルデータ

| カラム | 型 | 説明 |
|--------|-----|------|
| `image_id` | INTEGER FK | images.id への参照 |
| `model_name` | VARCHAR | モデル名 (例: "google/siglip2-base-patch16-224") |
| `embedding` | FLOAT[768] | 画像エンベディング (L2正規化済み) |

PK: (image_id, model_name)

#### `face_detections` テーブル — 顔検出

| カラム | 型 | 説明 |
|--------|-----|------|
| `face_id` | VARCHAR PK | 顔の一意 ID |
| `image_id` | INTEGER FK | images.id への参照 |
| `model_name` | VARCHAR | モデル名 ("insightface/buffalo_l") |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | FLOAT | バウンディングボックス |
| `det_score` | FLOAT | 検出スコア |
| `age` | INTEGER | 推定年齢 (nullable) |
| `gender` | VARCHAR | 性別 "M"/"F" (nullable) |
| `embedding` | FLOAT[512] | 顔エンベディング (ArcFace 512次元) |
| `person_label` | VARCHAR | 人物ラベル (nullable) |
| `cluster_id` | INTEGER | クラスタ ID (nullable) |

#### `object_detections` テーブル — 物体検出 (YOLO)

| カラム | 型 | 説明 |
|--------|-----|------|
| `detection_id` | VARCHAR PK | 検出の一意 ID |
| `image_id` | INTEGER FK | images.id への参照 |
| `model_name` | VARCHAR | モデル名 ("yolo11s") |
| `label` | VARCHAR | 物体ラベル ("person", "dog" 等) |
| `confidence` | FLOAT | 検出信頼度 (0-1) |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | FLOAT | バウンディングボックス |

### Firestore コレクション設計

DuckDB のデータを以下のコレクションにマッピングする。物体検出 (`object_detections`) は `images` ドキュメントの `tags` 配列に非正規化するため、個別コレクションとしては Firestore に格納しない。

#### コレクション: `images/{flickr_photo_id}`

`images` + `image_embeddings` (SigLIP 2 Base) を統合。

```
images/{flickr_photo_id}
  ├── image_url: string
  ├── flickr_photo_id: string
  ├── album_id: string
  ├── album_title: string
  ├── event_name: string
  ├── event_year: number
  ├── event_type: string
  ├── width: number
  ├── height: number
  ├── tags: string[]            ← object_detections のラベルを非正規化
  └── embedding: vector(768)   ← FieldValue.vector()
```

#### コレクション: `face_detections/{face_id}`

顔検出結果 + 顔エンベディング。

```
face_detections/{face_id}
  ├── flickr_photo_id: string      ← images への参照キー
  ├── event_name: string           ← images から非正規化（フィルタ用）
  ├── model_name: string           ("insightface/buffalo_l")
  ├── bbox_x1: number
  ├── bbox_y1: number
  ├── bbox_x2: number
  ├── bbox_y2: number
  ├── det_score: number
  ├── age: number | null
  ├── gender: string | null
  ├── person_label: string | null
  ├── cluster_id: number | null
  └── embedding: vector(512)       ← 顔エンベディング
```

#### コレクション: `metadata/filters`

フィルタ UI 用のメタデータ（DISTINCT の代替）。

```
metadata/filters
  ├── event_names: string[]    ← 全イベント名の一覧
  └── tag_labels: string[]     ← 全物体検出ラベルの一覧
```

### インデックス

`firestore.indexes.json` に定義。Firestore は `flat` (KNN 全探索) のみサポート。

| # | コレクション | フィールド | 用途 |
|---|-------------|-----------|------|
| 1 | `images` | `embedding` vec(768) | 画像ベクトル検索 |
| 2 | `images` | `event_name` + `embedding` vec(768) | イベントフィルタ付きベクトル検索 |
| 3 | `images` | `tags` (array) + `embedding` vec(768) | タグフィルタ付きベクトル検索 |
| 4 | `face_detections` | `embedding` vec(512) | 顔ベクトル検索 |
| 5 | `face_detections` | `event_name` + `embedding` vec(512) | イベントフィルタ付き顔検索 |

**注意**: Firestore は `in` と `array-contains-any` の同時使用不可。event_name + タグの複合フィルタはクライアント側で2段階処理する。

### Firestore へのデータインポート

DuckDB のデータを Firestore に投入するスクリプト: `scripts/upload_to_firestore.py`

#### 1. 認証セットアップ

gcloud CLI をインストール: https://cloud.google.com/sdk/docs/install

```bash
# Application Default Credentials でログイン
gcloud auth application-default login --project pyconjp-image-search
```

#### 2. データ投入

```bash
cd /path/to/pyconjp-image-search

# dry-run で投入件数を確認
uv run scripts/upload_to_firestore.py --dry-run

# 実際に投入
uv run scripts/upload_to_firestore.py
```

オプション:

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--db` | `pyconjp_image_search.duckdb` | DuckDB ファイルパス |
| `--project` | `pyconjp-image-search` | Firebase プロジェクト ID |
| `--dry-run` | — | 件数のみ表示、データは投入しない |
| `--force` | — | 既存ドキュメントをスキップせず上書きする |

スキーマ変更（フィールド追加等）後に既存データを更新する場合は `--force` を使用する。

#### 非正規化フィールド

データ投入時に以下の非正規化が行われる（Firestore は JOIN 不可のため）:

- **`images` → `tags`**: `object_detections` のラベルを画像ごとに集約した配列
- **`face_detections` → `event_name`**: 対応する画像の `event_name` をコピー
- **`metadata/filters`**: 全イベント名・全タグラベルの一覧（DISTINCT の代替）

#### 3. インデックスのデプロイ

```bash
cd web
npx firebase deploy --only firestore:indexes
```

Firebase コンソールでインデックスのステータスが `READY` になるまで待つ。

### Hosting の SPA 設定

`firebase.json` の `rewrites` で全リクエストを `/index.html` にリライトする SPA 設定が有効になっています。

### GitHub Actions に必要なシークレット

- `FIREBASE_SERVICE_ACCOUNT_PYCONJP_IMAGE_SEARCH`: Firebase サービスアカウントの認証情報 (GitHub リポジトリの Secrets に設定)
