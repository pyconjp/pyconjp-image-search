import type { AsyncDuckDBConnection } from "@duckdb/duckdb-wasm";
import type { FaceInfo, SearchResult } from "../types";

const MODEL_NAME = "openai/clip-vit-large-patch14";
const FACE_MODEL_NAME = "insightface/buffalo_l";

export async function searchByEmbedding(
  conn: AsyncDuckDBConnection,
  queryEmbedding: Float32Array,
  options: {
    limit: number;
    offset: number;
    eventNames?: string[];
  },
): Promise<SearchResult[]> {
  const vecStr = `[${Array.from(queryEmbedding).join(",")}]`;

  let whereClause = `e.model_name = '${MODEL_NAME}'`;
  if (options.eventNames && options.eventNames.length > 0) {
    const escaped = options.eventNames.map((n) => `'${n.replace(/'/g, "''")}'`);
    whereClause += ` AND i.event_name IN (${escaped.join(",")})`;
  }

  const sql = `
    SELECT
      i.id, i.image_url, i.event_name, i.event_year,
      i.album_title, i.flickr_photo_id,
      list_cosine_similarity(e.embedding, ${vecStr}::FLOAT[768]) AS score
    FROM data.image_embeddings e
    JOIN data.images i ON i.id = e.image_id
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

export async function getImageEmbedding(
  conn: AsyncDuckDBConnection,
  imageId: number,
): Promise<Float32Array | null> {
  const result = await conn.query(
    `SELECT embedding FROM data.image_embeddings WHERE image_id = ${imageId} AND model_name = '${MODEL_NAME}'`,
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
  },
): Promise<SearchResult[]> {
  const vecStr = `[${faceEmbedding.join(",")}]`;

  let whereClause = `f.model_name = '${FACE_MODEL_NAME}'`;
  if (options.eventNames && options.eventNames.length > 0) {
    const escaped = options.eventNames.map((n) => `'${n.replace(/'/g, "''")}'`);
    whereClause += ` AND i.event_name IN (${escaped.join(",")})`;
  }

  const sql = `
    SELECT
      i.id, i.image_url, i.event_name, i.event_year,
      i.album_title, i.flickr_photo_id,
      list_cosine_similarity(f.embedding, ${vecStr}::FLOAT[512]) AS score
    FROM data.face_detections f
    JOIN data.images i ON i.id = f.image_id
    WHERE ${whereClause}
    ORDER BY score DESC
    LIMIT ${options.limit}
    OFFSET ${options.offset}
  `;

  const result = await conn.query(sql);
  const rows = result.toArray();

  // Deduplicate by image_id (keep highest face score per image)
  const seen = new Map<number, SearchResult>();
  for (const row of rows) {
    const id = Number(row.id);
    const score = Number(row.score);
    const existing = seen.get(id);
    if (!existing || score > existing.score) {
      seen.set(id, {
        id,
        image_url: String(row.image_url),
        event_name: String(row.event_name),
        event_year: Number(row.event_year),
        album_title: row.album_title ? String(row.album_title) : null,
        flickr_photo_id: row.flickr_photo_id
          ? String(row.flickr_photo_id)
          : null,
        score,
      });
    }
  }

  return Array.from(seen.values()).sort((a, b) => b.score - a.score);
}
