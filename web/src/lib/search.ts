import type { AsyncDuckDBConnection } from "@duckdb/duckdb-wasm";
import type { FaceInfo, SearchResult } from "../types";

const FACE_MODEL_NAME = "insightface/buffalo_l";

export interface SearchConfig {
  modelName: string;
  embeddingDim: number;
}

export async function searchByEmbedding(
  conn: AsyncDuckDBConnection,
  queryEmbedding: Float32Array,
  options: {
    limit: number;
    offset: number;
    eventNames?: string[];
    tagNames?: string[];
  },
  config: SearchConfig,
): Promise<SearchResult[]> {
  const vecStr = `[${Array.from(queryEmbedding).join(",")}]`;

  let whereClause = `e.model_name = '${config.modelName}'`;
  if (options.eventNames && options.eventNames.length > 0) {
    const escaped = options.eventNames.map((n) => `'${n.replace(/'/g, "''")}'`);
    whereClause += ` AND i.event_name IN (${escaped.join(",")})`;
  }

  let tagJoin = "";
  if (options.tagNames && options.tagNames.length > 0) {
    const escapedTags = options.tagNames.map(
      (t) => `'${t.replace(/'/g, "''")}'`,
    );
    tagJoin = `JOIN (SELECT DISTINCT image_id FROM data.object_detections WHERE label IN (${escapedTags.join(",")})) od ON od.image_id = i.id`;
  }

  const sql = `
    SELECT
      i.id, i.image_url, i.event_name, i.event_year,
      i.album_title, i.flickr_photo_id,
      list_cosine_similarity(e.embedding, ${vecStr}::FLOAT[${config.embeddingDim}]) AS score
    FROM data.image_embeddings e
    JOIN data.images i ON i.id = e.image_id
    ${tagJoin}
    WHERE ${whereClause}
    ORDER BY score DESC
    LIMIT ${options.limit}
    OFFSET ${options.offset}
  `;

  const result = await conn.query(sql);
  const rows = result.toArray();
  return rows.map((row) => ({
    id: Number(row.id),
    image_url: String(row.image_url),
    event_name: String(row.event_name),
    event_year: Number(row.event_year),
    album_title: row.album_title ? String(row.album_title) : null,
    flickr_photo_id: row.flickr_photo_id ? String(row.flickr_photo_id) : null,
    score: Number(row.score),
  }));
}

export async function getEventNames(
  conn: AsyncDuckDBConnection,
): Promise<string[]> {
  const result = await conn.query(
    "SELECT DISTINCT event_name FROM data.images ORDER BY event_name",
  );
  return result.toArray().map((row) => String(row.event_name));
}

export async function getTagNames(
  conn: AsyncDuckDBConnection,
): Promise<string[]> {
  try {
    const result = await conn.query(
      "SELECT DISTINCT label FROM data.object_detections ORDER BY label",
    );
    return result.toArray().map((row) => String(row.label));
  } catch {
    return [];
  }
}

export async function getImageEmbedding(
  conn: AsyncDuckDBConnection,
  imageId: number,
  modelName: string,
): Promise<Float32Array | null> {
  const result = await conn.query(
    `SELECT embedding FROM data.image_embeddings WHERE image_id = ${imageId} AND model_name = '${modelName}'`,
  );
  const rows = result.toArray();
  if (rows.length === 0) return null;
  // DuckDB-WASM returns list columns as arrays
  const embArray = rows[0]!.embedding as number[];
  return new Float32Array(embArray);
}

export async function getFacesForImage(
  conn: AsyncDuckDBConnection,
  imageId: number,
): Promise<FaceInfo[]> {
  const result = await conn.query(`
    SELECT f.face_id, f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
           f.det_score, f.embedding, i.width, i.height
    FROM data.face_detections f
    JOIN data.images i ON i.id = f.image_id
    WHERE f.image_id = ${imageId} AND f.model_name = '${FACE_MODEL_NAME}'
    ORDER BY f.det_score DESC
  `);
  const rows = result.toArray();
  return rows.map((row) => ({
    face_id: String(row.face_id),
    bbox: [
      Number(row.bbox_x1),
      Number(row.bbox_y1),
      Number(row.bbox_x2),
      Number(row.bbox_y2),
    ] as [number, number, number, number],
    det_score: Number(row.det_score),
    embedding: Array.from(row.embedding as number[]),
    image_width: Number(row.width),
    image_height: Number(row.height),
  }));
}

export async function searchByFaceEmbedding(
  conn: AsyncDuckDBConnection,
  faceEmbedding: number[],
  options: {
    limit: number;
    offset: number;
    eventNames?: string[];
    tagNames?: string[];
  },
): Promise<SearchResult[]> {
  const vecStr = `[${faceEmbedding.join(",")}]`;

  let whereClause = `f.model_name = '${FACE_MODEL_NAME}'`;
  if (options.eventNames && options.eventNames.length > 0) {
    const escaped = options.eventNames.map((n) => `'${n.replace(/'/g, "''")}'`);
    whereClause += ` AND i.event_name IN (${escaped.join(",")})`;
  }

  let tagJoin = "";
  if (options.tagNames && options.tagNames.length > 0) {
    const escapedTags = options.tagNames.map(
      (t) => `'${t.replace(/'/g, "''")}'`,
    );
    tagJoin = `JOIN (SELECT DISTINCT image_id FROM data.object_detections WHERE label IN (${escapedTags.join(",")})) od ON od.image_id = i.id`;
  }

  const sql = `
    SELECT
      i.id, i.image_url, i.event_name, i.event_year,
      i.album_title, i.flickr_photo_id,
      MAX(list_cosine_similarity(f.embedding, ${vecStr}::FLOAT[512])) AS score
    FROM data.face_detections f
    JOIN data.images i ON i.id = f.image_id
    ${tagJoin}
    WHERE ${whereClause}
    GROUP BY i.id, i.image_url, i.event_name, i.event_year, i.album_title, i.flickr_photo_id
    ORDER BY score DESC
    LIMIT ${options.limit}
    OFFSET ${options.offset}
  `;

  const result = await conn.query(sql);
  const rows = result.toArray();
  return rows.map((row) => ({
    id: Number(row.id),
    image_url: String(row.image_url),
    event_name: String(row.event_name),
    event_year: Number(row.event_year),
    album_title: row.album_title ? String(row.album_title) : null,
    flickr_photo_id: row.flickr_photo_id ? String(row.flickr_photo_id) : null,
    score: Number(row.score),
  }));
}

export async function searchByMultipleFaceEmbeddings(
  conn: AsyncDuckDBConnection,
  faceEmbeddings: number[][],
  options: {
    limit: number;
    offset: number;
    eventNames?: string[];
    tagNames?: string[];
  },
): Promise<SearchResult[]> {
  if (faceEmbeddings.length === 1 && faceEmbeddings[0]) {
    return searchByFaceEmbedding(conn, faceEmbeddings[0], options);
  }

  const numFaces = faceEmbeddings.length;

  // Search each face independently with a much larger limit to get enough candidates
  const perFaceResults = await Promise.all(
    faceEmbeddings.map((emb) =>
      searchByFaceEmbedding(conn, emb, {
        limit: options.limit * 10,
        offset: 0,
        eventNames: options.eventNames,
      }),
    ),
  );

  // Build map: imageId -> scores for each face query
  const imageScoreMap = new Map<
    number,
    { result: SearchResult; scores: number[] }
  >();
  for (let fi = 0; fi < perFaceResults.length; fi++) {
    for (const result of perFaceResults[fi] ?? []) {
      let entry = imageScoreMap.get(result.id);
      if (!entry) {
        entry = {
          result,
          scores: new Array(numFaces).fill(-1),
        };
        imageScoreMap.set(result.id, entry);
      }
      entry.scores[fi] = result.score;
    }
  }

  // Score images: prioritize all-face matches, then allow partial matches
  const scored: { result: SearchResult; combinedScore: number }[] = [];
  for (const { result, scores } of imageScoreMap.values()) {
    const matchedCount = scores.filter((s) => s > 0).length;
    if (matchedCount === 0) continue;

    // Average of matched scores (treat unmatched as 0)
    const avgScore =
      scores.reduce((sum, s) => sum + Math.max(0, s), 0) / numFaces;
    // Bonus for matching more faces: multiply by (matchedCount / numFaces)
    // All faces matched -> 1.0x, partial -> proportional
    const matchRatio = matchedCount / numFaces;
    // Combined: weight heavily toward all-match but still show partial
    const combinedScore = avgScore * (0.5 + 0.5 * matchRatio);

    scored.push({ result: { ...result, score: combinedScore }, combinedScore });
  }

  scored.sort((a, b) => b.combinedScore - a.combinedScore);
  return scored.slice(0, options.limit).map((s) => s.result);
}
