import {
  collection,
  doc,
  getDoc,
  getDocs,
  query,
  where,
} from "firebase/firestore";
import type { FaceInfo, SearchResult } from "../types";
import type { DataSource, SearchOptions } from "./datasource";
import { auth, db } from "./firebase";
import { loadVoronoiCentroids, selectTopPartitions } from "./search";

/**
 * Firestore vector search via REST API.
 * The Firebase JS SDK does not expose `findNearest`, so we call the
 * Firestore REST `runQuery` endpoint directly with the authenticated user's token.
 */

interface RestDocField {
  stringValue?: string;
  integerValue?: string;
  doubleValue?: number;
  arrayValue?: { values?: RestDocField[] };
  mapValue?: { fields?: Record<string, RestDocField> };
  booleanValue?: boolean;
  nullValue?: null;
}

interface RestDocument {
  name: string;
  fields: Record<string, RestDocField>;
}

interface RunQueryResponse {
  document?: RestDocument;
}

function getProjectId(): string {
  return import.meta.env.VITE_FIREBASE_PROJECT_ID ?? "pyconjp-image-search";
}

async function getAuthToken(): Promise<string> {
  const user = auth.currentUser;
  if (user) {
    return user.getIdToken();
  }
  return "";
}

function extractStringField(
  fields: Record<string, RestDocField>,
  key: string,
): string {
  return fields[key]?.stringValue ?? "";
}

function extractNumberField(
  fields: Record<string, RestDocField>,
  key: string,
): number {
  if (fields[key]?.integerValue) return Number(fields[key].integerValue);
  if (fields[key]?.doubleValue != null) return fields[key].doubleValue ?? 0;
  return 0;
}

/**
 * Convert Firestore VectorValue or plain array to number[].
 * Firestore SDK returns Vector fields as VectorValue objects with a toArray() method.
 */
function toNumberArray(value: unknown): number[] {
  if (Array.isArray(value)) return value as number[];
  if (value && typeof value === "object" && "toArray" in value) {
    return (value as { toArray(): number[] }).toArray();
  }
  return [];
}

function extractDocId(name: string): string {
  // name format: projects/{proj}/databases/{db}/documents/{collection}/{docId}
  return name.split("/").pop() ?? "";
}

function buildVectorQueryBody(
  collectionId: string,
  vectorField: string,
  queryVector: number[],
  limit: number,
  filters?: { field: string; op: string; value: unknown }[],
  distanceResultField?: string,
): Record<string, unknown> {
  const structuredQuery: Record<string, unknown> = {
    from: [{ collectionId }],
    findNearest: {
      vectorField: { fieldPath: vectorField },
      queryVector: {
        mapValue: {
          fields: {
            __type__: { stringValue: "__vector__" },
            value: {
              arrayValue: {
                values: queryVector.map((v) => ({ doubleValue: v })),
              },
            },
          },
        },
      },
      distanceMeasure: "COSINE",
      limit: { value: limit },
      ...(distanceResultField ? { distanceResultField } : {}),
    },
  };

  // Build composite filter if needed
  if (filters && filters.length > 0) {
    const firestoreFilters = filters.map((f) => {
      let value: Record<string, unknown>;
      if (f.op === "IN") {
        // For IN, value is an array
        value = {
          arrayValue: {
            values: (f.value as string[]).map((v) => ({ stringValue: v })),
          },
        };
      } else if (f.op === "ARRAY_CONTAINS_ANY") {
        const arr = f.value as (string | number)[];
        value = {
          arrayValue: {
            values: arr.map((v) =>
              typeof v === "number"
                ? { integerValue: String(v) }
                : { stringValue: v },
            ),
          },
        };
      } else {
        value = { stringValue: f.value as string };
      }
      return {
        fieldFilter: {
          field: { fieldPath: f.field },
          op: f.op,
          value,
        },
      };
    });

    if (firestoreFilters.length === 1) {
      structuredQuery.where = firestoreFilters[0];
    } else {
      structuredQuery.where = {
        compositeFilter: {
          op: "AND",
          filters: firestoreFilters,
        },
      };
    }
  }

  return { structuredQuery };
}

async function runVectorQuery(
  collectionId: string,
  vectorField: string,
  queryVector: number[],
  limit: number,
  filters?: { field: string; op: string; value: unknown }[],
): Promise<RestDocument[]> {
  const projectId = getProjectId();
  const token = await getAuthToken();
  const body = buildVectorQueryBody(
    collectionId,
    vectorField,
    queryVector,
    limit,
    filters,
    "vector_distance",
  );

  const url = `https://firestore.googleapis.com/v1/projects/${projectId}/databases/(default)/documents:runQuery`;
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    if (resp.status === 429) {
      throw new Error(
        "Firestoreのクォータ上限に達しました。しばらく時間をおいてから再度お試しください。",
      );
    }
    const text = await resp.text();
    throw new Error(`Firestore query failed: ${resp.status} ${text}`);
  }

  const results: RunQueryResponse[] = await resp.json();
  return results
    .filter(
      (r): r is RunQueryResponse & { document: RestDocument } =>
        r.document != null,
    )
    .map((r) => r.document);
}

export class FirestoreDataSource implements DataSource {
  async searchByEmbedding(
    queryEmbedding: Float32Array,
    options: SearchOptions,
  ): Promise<SearchResult[]> {
    const filters: { field: string; op: string; value: unknown }[] = [];

    if (options.eventNames && options.eventNames.length > 0) {
      filters.push({
        field: "event_name",
        op: "IN",
        value: options.eventNames.slice(0, 30),
      });
    }

    // Cannot combine IN and ARRAY_CONTAINS_ANY in the same query
    if (
      options.tagNames &&
      options.tagNames.length > 0 &&
      !(options.eventNames && options.eventNames.length > 0)
    ) {
      filters.push({
        field: "tags",
        op: "ARRAY_CONTAINS_ANY",
        value: options.tagNames.slice(0, 30),
      });
    }

    const docs = await runVectorQuery(
      "images",
      "embedding",
      Array.from(queryEmbedding),
      options.limit + options.offset,
      filters.length > 0 ? filters : undefined,
    );

    let results: SearchResult[] = docs.map((d) => ({
      id: 0,
      image_url: extractStringField(d.fields, "image_url"),
      event_name: extractStringField(d.fields, "event_name"),
      event_year: extractNumberField(d.fields, "event_year"),
      album_title: extractStringField(d.fields, "album_title") || null,
      flickr_photo_id: extractDocId(d.name),
      score: 1 - extractNumberField(d.fields, "vector_distance"), // cosine distance -> similarity
    }));

    // Client-side tag filter when both event and tag filters are active
    if (
      options.tagNames &&
      options.tagNames.length > 0 &&
      options.eventNames &&
      options.eventNames.length > 0
    ) {
      const tagSet = new Set(options.tagNames);
      results = results.filter((r) => {
        const doc = docs.find(
          (d) => extractDocId(d.name) === r.flickr_photo_id,
        );
        const tagsField = doc?.fields.tags?.arrayValue?.values;
        if (!tagsField) return false;
        return tagsField.some(
          (v) => v.stringValue && tagSet.has(v.stringValue),
        );
      });
    }

    return results.slice(options.offset);
  }

  async searchByFaceEmbedding(
    faceEmbedding: number[],
    options: SearchOptions,
  ): Promise<SearchResult[]> {
    const filters: { field: string; op: string; value: unknown }[] = [];
    let clientEventFilter: Set<string> | null = null;

    // Voronoi pre-filtering: use ARRAY_CONTAINS_ANY on voronoi_partition_ids
    const useVoronoi = options.useVoronoi !== false;
    if (useVoronoi) {
      const centroids = await loadVoronoiCentroids();
      if (centroids.length > 0) {
        const partitionIds = selectTopPartitions(faceEmbedding, centroids, 5);
        filters.push({
          field: "voronoi_partition_ids",
          op: "ARRAY_CONTAINS_ANY",
          value: partitionIds,
        });
        // Firestore cannot combine IN + ARRAY_CONTAINS_ANY,
        // so event_name filter must be done client-side
        if (options.eventNames && options.eventNames.length > 0) {
          clientEventFilter = new Set(options.eventNames);
        }
      } else {
        // Fallback: no centroids available, use event filter in query
        if (options.eventNames && options.eventNames.length > 0) {
          filters.push({
            field: "event_name",
            op: "IN",
            value: options.eventNames.slice(0, 30),
          });
        }
      }
    } else {
      // Full scan mode: no Voronoi filter, use event filter in query
      if (options.eventNames && options.eventNames.length > 0) {
        filters.push({
          field: "event_name",
          op: "IN",
          value: options.eventNames.slice(0, 30),
        });
      }
    }

    // Request more results when client-side filtering is needed
    const queryLimit = clientEventFilter
      ? (options.limit + options.offset) * 5
      : options.limit + options.offset;

    const docs = await runVectorQuery(
      "face_detections",
      "embedding",
      faceEmbedding,
      queryLimit,
      filters.length > 0 ? filters : undefined,
    );

    // Group by flickr_photo_id, keeping the first (closest) occurrence
    const imageMap = new Map<
      string,
      { fields: Record<string, RestDocField>; distance: number }
    >();
    for (const d of docs) {
      const photoId = extractStringField(d.fields, "flickr_photo_id");
      if (!photoId || imageMap.has(photoId)) continue;
      const distance = extractNumberField(d.fields, "vector_distance");
      imageMap.set(photoId, { fields: d.fields, distance });
    }

    // Fetch image details for the found photo IDs
    const photoIds = Array.from(imageMap.keys());
    const results: SearchResult[] = [];

    await Promise.all(
      photoIds.map(async (photoId) => {
        const imgDoc = await getDoc(doc(db, "images", photoId));
        if (imgDoc.exists()) {
          const imgData = imgDoc.data();
          const entry = imageMap.get(photoId);
          const score = entry ? 1 - entry.distance : 0;
          results.push({
            id: 0,
            image_url: imgData.image_url as string,
            event_name: imgData.event_name as string,
            event_year: imgData.event_year as number,
            album_title: (imgData.album_title as string) ?? null,
            flickr_photo_id: photoId,
            score,
          });
        }
      }),
    );

    // Preserve order from vector search
    let orderedResults: SearchResult[] = [];
    for (const photoId of photoIds) {
      const r = results.find((r) => r.flickr_photo_id === photoId);
      if (r) orderedResults.push(r);
    }

    // Client-side event filter when Voronoi + event filter are both active
    if (clientEventFilter) {
      const filterSet = clientEventFilter;
      orderedResults = orderedResults.filter((r) =>
        filterSet.has(r.event_name),
      );
    }

    return orderedResults.slice(options.offset, options.offset + options.limit);
  }

  async searchByMultipleFaceEmbeddings(
    faceEmbeddings: number[][],
    options: SearchOptions,
  ): Promise<SearchResult[]> {
    if (faceEmbeddings.length === 1 && faceEmbeddings[0]) {
      return this.searchByFaceEmbedding(faceEmbeddings[0], options);
    }

    const perFaceResults = await Promise.all(
      faceEmbeddings.map((emb) =>
        this.searchByFaceEmbedding(emb, {
          limit: options.limit * 10,
          offset: 0,
          eventNames: options.eventNames,
          tagNames: options.tagNames,
        }),
      ),
    );

    const imageMap = new Map<
      string,
      { result: SearchResult; matchCount: number }
    >();
    for (const results of perFaceResults) {
      const seen = new Set<string>();
      for (const result of results) {
        const key = result.flickr_photo_id ?? "";
        if (seen.has(key)) continue;
        seen.add(key);
        const entry = imageMap.get(key);
        if (entry) {
          entry.matchCount++;
        } else {
          imageMap.set(key, { result, matchCount: 1 });
        }
      }
    }

    const scored = Array.from(imageMap.values());
    scored.sort((a, b) => b.matchCount - a.matchCount);
    return scored.slice(0, options.limit).map((s) => s.result);
  }

  async getEventNames(): Promise<string[]> {
    const docSnap = await getDoc(doc(db, "metadata", "filters"));
    if (!docSnap.exists()) return [];
    return (docSnap.data().event_names as string[]) ?? [];
  }

  async getTagNames(): Promise<string[]> {
    const docSnap = await getDoc(doc(db, "metadata", "filters"));
    if (!docSnap.exists()) return [];
    return (docSnap.data().tag_labels as string[]) ?? [];
  }

  async getFacesForImage(
    _imageId: number,
    flickrPhotoId?: string,
  ): Promise<FaceInfo[]> {
    if (!flickrPhotoId) return [];
    const q = query(
      collection(db, "face_detections"),
      where("flickr_photo_id", "==", flickrPhotoId),
    );
    const snapshot = await getDocs(q);

    const imgDoc = await getDoc(doc(db, "images", flickrPhotoId));
    const imgData = imgDoc.exists() ? imgDoc.data() : null;
    const width = (imgData?.width as number) ?? 0;
    const height = (imgData?.height as number) ?? 0;

    return snapshot.docs
      .map((docSnap) => {
        const data = docSnap.data();
        if (!data.embedding) return null;
        return {
          face_id: docSnap.id,
          bbox: [
            data.bbox_x1 as number,
            data.bbox_y1 as number,
            data.bbox_x2 as number,
            data.bbox_y2 as number,
          ] as [number, number, number, number],
          det_score: data.det_score as number,
          embedding: toNumberArray(data.embedding),
          image_width: width,
          image_height: height,
        };
      })
      .filter((f): f is FaceInfo => f != null);
  }

  async getImageEmbedding(
    _imageId: number,
    flickrPhotoId?: string,
  ): Promise<Float32Array | null> {
    if (!flickrPhotoId) return null;
    const docSnap = await getDoc(doc(db, "images", flickrPhotoId));
    if (!docSnap.exists()) return null;
    const embedding = docSnap.data().embedding;
    if (!embedding) return null;
    return new Float32Array(toNumberArray(embedding));
  }
}

/**
 * Extended Firestore datasource that can look up by flickr_photo_id.
 * Overrides getFacesForImage and getImageEmbedding are still no-ops
 * for numeric IDs, but provides methods for flickr_photo_id-based lookup.
 */
export class FirestoreDataSourceExtended extends FirestoreDataSource {
  async getFacesForFlickrPhoto(flickrPhotoId: string): Promise<FaceInfo[]> {
    const q = query(
      collection(db, "face_detections"),
      where("flickr_photo_id", "==", flickrPhotoId),
    );
    const snapshot = await getDocs(q);

    // Also fetch the image doc for width/height
    const imgDoc = await getDoc(doc(db, "images", flickrPhotoId));
    const imgData = imgDoc.exists() ? imgDoc.data() : null;
    const width = (imgData?.width as number) ?? 0;
    const height = (imgData?.height as number) ?? 0;

    return snapshot.docs
      .map((docSnap) => {
        const data = docSnap.data();
        if (!data.embedding) return null;
        return {
          face_id: docSnap.id,
          bbox: [
            data.bbox_x1 as number,
            data.bbox_y1 as number,
            data.bbox_x2 as number,
            data.bbox_y2 as number,
          ] as [number, number, number, number],
          det_score: data.det_score as number,
          embedding: toNumberArray(data.embedding),
          image_width: width,
          image_height: height,
        };
      })
      .filter((f): f is FaceInfo => f != null);
  }

  async getImageEmbeddingByFlickrPhoto(
    flickrPhotoId: string,
  ): Promise<Float32Array | null> {
    const docSnap = await getDoc(doc(db, "images", flickrPhotoId));
    if (!docSnap.exists()) return null;
    const embedding = docSnap.data().embedding;
    if (!embedding) return null;
    return new Float32Array(toNumberArray(embedding));
  }
}
