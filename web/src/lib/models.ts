export interface ModelConfig {
  /** Internal identifier */
  id: string;
  /** Display label */
  label: string;
  /** HuggingFace model ID for Transformers.js (ONNX) */
  onnxModelId: string;
  /** Model type determines which encoder classes to use */
  modelType: "siglip" | "clip";
  /** DuckDB file name served from /public */
  dbFileName: string;
  /** Model name stored in the DB's image_embeddings.model_name column */
  modelName: string;
  /** Embedding vector dimension */
  embeddingDim: number;
  /** Whether to use FP16 dtype for ONNX model loading */
  useFp16: boolean;
}

export const DEFAULT_CONFIG: ModelConfig = {
  id: "siglip2-base",
  label: "SigLIP 2 base",
  onnxModelId: "onnx-community/siglip2-base-patch16-224-ONNX",
  modelType: "siglip",
  dbFileName: "pyconjp_image_search.duckdb",
  modelName: "google/siglip2-base-patch16-224",
  embeddingDim: 768,
  useFp16: true,
};
