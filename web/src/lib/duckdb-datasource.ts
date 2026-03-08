import type { AsyncDuckDBConnection } from "@duckdb/duckdb-wasm";
import type { DataSource, SearchOptions } from "./datasource";
import type { ModelConfig } from "./models";
import {
  type SearchConfig,
  getEventNames as duckGetEventNames,
  getFacesForImage as duckGetFacesForImage,
  getImageEmbedding as duckGetImageEmbedding,
  getTagNames as duckGetTagNames,
  searchByEmbedding as duckSearchByEmbedding,
  searchByFaceEmbedding as duckSearchByFaceEmbedding,
  searchByMultipleFaceEmbeddings as duckSearchByMultipleFaceEmbeddings,
} from "./search";

export class DuckDBDataSource implements DataSource {
  constructor(
    private conn: AsyncDuckDBConnection,
    private config: ModelConfig,
  ) {}

  private get searchConfig(): SearchConfig {
    return {
      modelName: this.config.modelName,
      embeddingDim: this.config.embeddingDim,
    };
  }

  async searchByEmbedding(
    queryEmbedding: Float32Array,
    options: SearchOptions,
  ) {
    return duckSearchByEmbedding(
      this.conn,
      queryEmbedding,
      options,
      this.searchConfig,
    );
  }

  async searchByFaceEmbedding(
    faceEmbedding: number[],
    options: SearchOptions,
  ) {
    return duckSearchByFaceEmbedding(this.conn, faceEmbedding, options);
  }

  async searchByMultipleFaceEmbeddings(
    faceEmbeddings: number[][],
    options: SearchOptions,
  ) {
    return duckSearchByMultipleFaceEmbeddings(
      this.conn,
      faceEmbeddings,
      options,
    );
  }

  async getEventNames() {
    return duckGetEventNames(this.conn);
  }

  async getTagNames() {
    return duckGetTagNames(this.conn);
  }

  async getFacesForImage(imageId: number) {
    return duckGetFacesForImage(this.conn, imageId);
  }

  async getImageEmbedding(imageId: number) {
    return duckGetImageEmbedding(this.conn, imageId, this.config.modelName);
  }
}
