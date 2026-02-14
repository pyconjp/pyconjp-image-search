import type { AsyncDuckDBConnection } from "@duckdb/duckdb-wasm";
import type { SearchResult } from "../types";

const MODEL_NAME = "openai/clip-vit-large-patch14";

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
