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

export const MODEL_CONFIGS: ModelConfig[] = [
  {
    id: "siglip2-base",
    label: "SigLIP 2 base",
    onnxModelId: "onnx-community/siglip2-base-patch16-224-ONNX",
    modelType: "siglip",
    dbFileName: "pyconjp_image_search.duckdb",
    modelName: "google/siglip2-base-patch16-224",
    embeddingDim: 768,
    useFp16: true,
  },
  {
    id: "siglip2-large",
    label: "SigLIP 2 Large",
    onnxModelId: "onnx-community/siglip2-large-patch16-256-ONNX",
    modelType: "siglip",
    dbFileName: "pyconjp_image_search_siglip2_large.duckdb",
    modelName: "google/siglip2-large-patch16-256",
    embeddingDim: 1024,
    useFp16: true,
  },
  {
    id: "clip-l",
    label: "CLIP-L",
    onnxModelId: "Xenova/clip-vit-large-patch14",
    modelType: "clip",
    dbFileName: "pyconjp_image_search_clip.duckdb",
    modelName: "openai/clip-vit-large-patch14",
    embeddingDim: 768,
    useFp16: false,
  },
];

const STORAGE_KEY = "pyconjp-model-selection";

export function getStoredModelId(): string | null {
  return localStorage.getItem(STORAGE_KEY);
}

export function storeModelId(id: string): void {
  localStorage.setItem(STORAGE_KEY, id);
}

export function clearStoredModelId(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export function getModelConfig(id: string): ModelConfig {
  return MODEL_CONFIGS.find((m) => m.id === id) ?? MODEL_CONFIGS[0]!;
}
