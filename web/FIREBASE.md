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

### Hosting の SPA 設定

`firebase.json` の `rewrites` で全リクエストを `/index.html` にリライトする SPA 設定が有効になっています。

### GitHub Actions に必要なシークレット

- `FIREBASE_SERVICE_ACCOUNT_PYCONJP_IMAGE_SEARCH`: Firebase サービスアカウントの認証情報 (GitHub リポジトリの Secrets に設定)
