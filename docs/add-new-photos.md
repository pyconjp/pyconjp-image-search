# 新しい写真の追加手順

Flickr に新しいアルバム（例: 新しい年の PyCon JP）が追加されたときに、ダウンロードから Firestore への反映までを行う手順です。

各ステップは**未処理のデータだけを処理する**ため、途中で失敗しても同じコマンドを再実行すれば続きから処理されます。

## 対象の DB

Firestore に投入されるのは **SigLIP 2 base の DB (`pyconjp_image_search.duckdb`)** のデータです。以下の手順はすべてこの DB に対して行います。

CLIP / SigLIP 2 Large の DB はこの手順では更新されません。ブラウザ内の DuckDB WASM でこれらを使う場合は、別途メタデータのコピーと Embedding 生成が必要です。

## 事前準備

```bash
uv sync --extra ml
gcloud auth application-default login --project pyconjp-image-search
```

`.env` に `FLICKR_API_KEY` と `FLICKR_USER_ID` が設定されていることを確認してください。公開アルバムを読むだけなので、Flickr へのログイン（OAuth）は不要です。

## 手順

### 1. 画像のダウンロード

```bash
uv run pyconjp-manage list-albums    # 新しいアルバムを確認
uv run scripts/download_all.py       # 全アルバムを差分ダウンロード
```

イベント名と年はアルバム名から自動で決まります（例: `PyCon JP 2026 Conference Day1` → `PyCon JP 2026` / 2026）。

ダウンロード後、イベント名に表記ゆれがないかを確認します。既存のアルバムに写真が追加されている場合もあり、古い表記（例: `Pycon JP 2019`）で登録されることがあります。

```bash
uv run pyconjp-manage list --year 2026
```

> **注意**: `scripts/fix_event_names.py` は使わないでください。`image_embeddings` を Voronoi 列のない古い定義で作り直すため、現在の DB を壊します。

### 2. Embedding・顔検出・物体検出

```bash
uv run pyconjp-embed generate --batch-size 32
uv run pyconjp-embed face-generate
uv run pyconjp-embed object-generate --primary-only
```

`object-generate` には `--primary-only` を付けてください。付けないと結果を他の DB にもコピーしようとします。他の DB には新しい画像が入っていないので、外部キーのエラーで止まります。

完了の確認:

```bash
uv run pyconjp-embed status
uv run pyconjp-embed face-status
```

どちらも `Processed`（または `Embedded`）が全件数と一致していれば OK です。

### 3. Voronoi パーティションの割り当て

```bash
uv run scripts/assign_image_voronoi_partitions.py --force
uv run scripts/assign_voronoi_partitions.py --force
```

**`--force` が必須です。** これらのスクリプトは、1件でも割り当て済みのデータがあると何もせずに終了します。`--force` を付けると全件を計算し直しますが、ローカル処理なので費用はかかりません。中心点（`data/*_pivots.json`）が同じなので、既存データの値も変わりません。

### 4. Firestore へのアップロード

```bash
uv run scripts/upload_to_firestore.py --dry-run
uv run scripts/upload_to_firestore.py
```

既存のドキュメントはスキップされ、新しい分だけが書き込まれます。画像・顔ともに `voronoi_partition_ids` 付きで登録され、`metadata/filters`（イベント名・タグ一覧）も更新されます。

**手順 3 より前に実行しないでください。** 新しい画像が Voronoi ID なしで登録され、Voronoi を使った検索で見つからなくなります。

> **費用に関する注意**: 新しい分を判定するために、既存のドキュメント ID を全件読み込みます（2026 年追加時点で画像約 2.3 万件、顔約 7.3 万件）。1 回の実行で Firestore の無料枠（読み取り 5 万回/日）を超えることがあります。

## トラブルシューティング

| 症状 | 原因と対処 |
|------|-----------|
| Flickr API で `502 Bad Gateway` | User-Agent が httpx のデフォルトのまま。`flickr_client.py` の `HTTP_HEADERS` が使われているか確認する |
| `face-generate` で `Unable to import dependency onnxruntime` | `uv sync --extra ml` を実行する（`onnxruntime-gpu` は ml extras に含まれている） |
| 顔検出が遅い（CPU で動いている） | `onnxruntime-gpu` の CUDA バージョンが torch と合っていない。torch が CUDA 12 系の間は `<1.24` を維持する |
| `face-generate` の `Errors` が処理件数と同じ | DB への書き込みに失敗している。face_id はランダムな UUID なので、再実行する前に処理済みでない画像の `face_detections` を削除しておく（削除しないと二重登録になる） |
| `upload_to_firestore.py` で quota exceeded | 無料枠の超過。翌日に再実行すれば続きから処理される |
