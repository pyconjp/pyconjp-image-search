import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  type PreTrainedTokenizer,
  type Processor,
  RawImage,
  SiglipTextModel,
  SiglipVisionModel,
} from "@huggingface/transformers";
import type { ModelConfig } from "./models";

function normalize(vec: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i]! * vec[i]!;
  norm = Math.sqrt(norm);
  if (norm < 1e-8) return vec;
  const result = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) result[i] = vec[i]! / norm;
  return result;
}

/**
 * Unified encoder supporting both CLIP and SigLIP 2 models via Transformers.js.
 */
export class VisionLanguageEncoder {
  private config: ModelConfig;
  private tokenizer: PreTrainedTokenizer | null = null;
  // biome-ignore lint/suspicious/noExplicitAny: model types vary by model family
  private textModel: any = null;
  private processor: Processor | null = null;
  // biome-ignore lint/suspicious/noExplicitAny: model types vary by model family
  private visionModel: any = null;

  constructor(config: ModelConfig) {
    this.config = config;
  }

  /** Load the text encoder (tokenizer + text model). */
  async loadTextModel(onProgress?: (progress: number) => void): Promise<void> {
    if (this.tokenizer && this.textModel) return;
    const modelId = this.config.onnxModelId;
    const progressCallback = onProgress
      ? (p: Record<string, unknown>) => {
          const prog = p.progress;
          if (typeof prog === "number") onProgress(prog);
        }
      : undefined;

    const opts = {
      progress_callback: progressCallback as never,
      ...(this.config.useFp16 ? { dtype: "fp16" as const } : {}),
    };

    if (this.config.modelType === "clip") {
      const [tokenizer, textModel] = await Promise.all([
        AutoTokenizer.from_pretrained(modelId, opts),
        CLIPTextModelWithProjection.from_pretrained(modelId, opts),
      ]);
      this.tokenizer = tokenizer;
      this.textModel = textModel;
    } else {
      // SigLIP 2
      const [tokenizer, textModel] = await Promise.all([
        AutoTokenizer.from_pretrained(modelId, opts),
        SiglipTextModel.from_pretrained(modelId, opts),
      ]);
      this.tokenizer = tokenizer;
      this.textModel = textModel;
    }
  }

  /** Load the vision encoder (processor + vision model). Lazy-loaded on first use. */
  async loadVisionModel(
    onProgress?: (progress: number) => void,
  ): Promise<void> {
    if (this.processor && this.visionModel) return;
    const modelId = this.config.onnxModelId;
    const progressCallback = onProgress
      ? (p: Record<string, unknown>) => {
          const prog = p.progress;
          if (typeof prog === "number") onProgress(prog);
        }
      : undefined;

    const opts = {
      progress_callback: progressCallback as never,
      ...(this.config.useFp16 ? { dtype: "fp16" as const } : {}),
    };

    if (this.config.modelType === "clip") {
      const [processor, visionModel] = await Promise.all([
        AutoProcessor.from_pretrained(modelId, {
          progress_callback: progressCallback as never,
        }),
        CLIPVisionModelWithProjection.from_pretrained(modelId, opts),
      ]);
      this.processor = processor;
      this.visionModel = visionModel;
    } else {
      // SigLIP 2
      const [processor, visionModel] = await Promise.all([
        AutoProcessor.from_pretrained(modelId, {
          progress_callback: progressCallback as never,
        }),
        SiglipVisionModel.from_pretrained(modelId, opts),
      ]);
      this.processor = processor;
      this.visionModel = visionModel;
    }
  }

  get isTextReady(): boolean {
    return this.tokenizer !== null && this.textModel !== null;
  }

  get isVisionReady(): boolean {
    return this.processor !== null && this.visionModel !== null;
  }

  async encodeText(text: string): Promise<Float32Array> {
    if (!this.tokenizer || !this.textModel) {
      throw new Error("Text model not loaded. Call loadTextModel() first.");
    }
    const inputs =
      this.config.modelType === "clip"
        ? this.tokenizer(text, { padding: true, truncation: true })
        : this.tokenizer(text, {
            padding: "max_length",
            max_length: 64,
            truncation: true,
          });

    if (this.config.modelType === "clip") {
      const { text_embeds } = await this.textModel(inputs);
      return normalize(text_embeds.data as Float32Array);
    }
    // SigLIP 2: SiglipTextModel outputs pooler_output
    const output = await this.textModel(inputs);
    const embedding = (output.pooler_output ?? output.last_hidden_state)
      .data as Float32Array;
    // For pooler_output, take the first embeddingDim elements (single text)
    const vec = embedding.slice(0, this.config.embeddingDim);
    return normalize(vec);
  }

  async encodeImage(imageBlob: Blob): Promise<Float32Array> {
    if (!this.processor || !this.visionModel) {
      throw new Error("Vision model not loaded. Call loadVisionModel() first.");
    }
    const image = await RawImage.fromBlob(imageBlob);
    const inputs = await this.processor(image);

    if (this.config.modelType === "clip") {
      const { image_embeds } = await this.visionModel(inputs);
      return normalize(image_embeds.data as Float32Array);
    }
    // SigLIP 2: SiglipVisionModel outputs pooler_output
    const output = await this.visionModel(inputs);
    const embedding = (output.pooler_output ?? output.last_hidden_state)
      .data as Float32Array;
    const vec = embedding.slice(0, this.config.embeddingDim);
    return normalize(vec);
  }
}
