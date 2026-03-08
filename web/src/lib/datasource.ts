import type { FaceInfo, SearchResult } from "../types";

export interface SearchOptions {
  limit: number;
  offset: number;
  eventNames?: string[];
  tagNames?: string[];
}

export interface DataSource {
  searchByEmbedding(
    queryEmbedding: Float32Array,
    options: SearchOptions,
  ): Promise<SearchResult[]>;

  searchByFaceEmbedding(
    faceEmbedding: number[],
    options: SearchOptions,
  ): Promise<SearchResult[]>;

  searchByMultipleFaceEmbeddings(
    faceEmbeddings: number[][],
    options: SearchOptions,
  ): Promise<SearchResult[]>;

  getEventNames(): Promise<string[]>;

  getTagNames(): Promise<string[]>;

  getFacesForImage(imageId: number): Promise<FaceInfo[]>;

  getImageEmbedding(imageId: number): Promise<Float32Array | null>;
}
