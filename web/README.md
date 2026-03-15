# PyCon JP Image Search - Web アプリ

React + Vite で構築されたクライアントサイド画像検索アプリケーションです。Firebase Hosting でデプロイされ、Firebase Auth によるアクセス制御が可能です。

## データソース

| モード | 説明 | 設定 |
|--------|------|------|
| **Firestore** (本番) | Firestore に保存されたデータを使用 | `VITE_DATASOURCE=firestore` |
| **DuckDB** (開発) | ブラウザ内 DuckDB WASM でローカル DB を読み込み | `VITE_DATASOURCE=duckdb` |

## セットアップ

### 環境変数

```bash
cp .env.example .env
```

`.env` に Firebase プロジェクトの設定を記入してください。`.env.example` を参照。

### 開発サーバー

```bash
npm install
npm run dev
```

http://localhost:5173 で起動します。

### 本番ビルド

```bash
npm run build
npm run preview
```

## 機能

- **テキスト検索** -- 検索テキストをブラウザ内で CLIP Embedding に変換し、類似画像を検索
- **英訳ボタン** -- Chrome Translator API (Chrome 138+) による日本語→英語翻訳。CLIP-L は英語に最適化されているため、英訳してから検索すると精度が向上
- **画像検索** -- 画像アップロードまたはクリップボードからの貼り付けで類似画像を検索
- **顔検索** -- 顔サムネイルクリックで同一人物を検索（Voronoi フィルタによる高速検索対応）
- **イベントフィルター** -- イベント名で絞り込み
- **物体タグフィルター** -- YOLO11 で検出された物体ラベル（person, laptop 等）で絞り込み
- **プレビュー** -- 画像クリックで拡大表示、サムネイルストリップ付き
- **Find Similar** -- 検索結果の画像から類似画像を再検索
- **クロップ検索** -- プレビュー画像上でドラッグして矩形選択し、その領域で類似検索

## アーキテクチャ

| 用途 | ライブラリ |
|------|-----------|
| フレームワーク | React 19 + TypeScript |
| ビルド | Vite 6 |
| Embedding | Transformers.js (`@huggingface/transformers`) |
| DB (DuckDB モード) | DuckDB WASM (`@duckdb/duckdb-wasm`) |
| DB (Firestore モード) | Firebase SDK |
| 認証 | Firebase Auth (Google Sign-In) |
| 翻訳 | Chrome Translator API (Chrome 138+) |
| ホスティング | Firebase Hosting |

## デプロイ

- **PR 時**: Firebase Hosting のプレビューチャンネルに自動デプロイ
- **main マージ時**: 本番 (`live`) に自動デプロイ

Firebase / Firestore の詳細設定は [FIREBASE.md](FIREBASE.md) を参照してください。
