import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  type PreTrainedTokenizer,
  type Processor,
  RawImage,
} from "@huggingface/transformers";

const MODEL_ID = "Xenova/clip-vit-large-patch14";

function normalize(vec: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i]! * vec[i]!;
  norm = Math.sqrt(norm);
  if (norm < 1e-8) return vec;
  const result = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) result[i] = vec[i]! / norm;
  return result;
}

export class CLIPEncoder {
  private tokenizer: PreTrainedTokenizer | null = null;
  private textModel: CLIPTextModelWithProjection | null = null;
  private processor: Processor | null = null;
  private visionModel: CLIPVisionModelWithProjection | null = null;

  /** Load the text encoder (tokenizer + text model). */
  async loadTextModel(onProgress?: (progress: number) => void): Promise<void> {
    if (this.tokenizer && this.textModel) return;
    const progressCallback = onProgress
      ? (p: Record<string, unknown>) => {
          const prog = p.progress;
          if (typeof prog === "number") onProgress(prog);
        }
      : undefined;
    const [tokenizer, textModel] = await Promise.all([
      AutoTokenizer.from_pretrained(MODEL_ID, {
        progress_callback: progressCallback as never,
      }),
      CLIPTextModelWithProjection.from_pretrained(MODEL_ID, {
        progress_callback: progressCallback as never,
      }),
    ]);
    this.tokenizer = tokenizer;
    this.textModel = textModel as CLIPTextModelWithProjection;
  }

  /** Load the vision encoder (processor + vision model). Lazy-loaded on first use. */
  async loadVisionModel(
    onProgress?: (progress: number) => void,
  ): Promise<void> {
    if (this.processor && this.visionModel) return;
    const progressCallback = onProgress
      ? (p: Record<string, unknown>) => {
          const prog = p.progress;
          if (typeof prog === "number") onProgress(prog);
        }
      : undefined;
    const [processor, visionModel] = await Promise.all([
      AutoProcessor.from_pretrained(MODEL_ID, {
        progress_callback: progressCallback as never,
      }),
      CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, {
        progress_callback: progressCallback as never,
      }),
    ]);
    this.processor = processor;
    this.visionModel = visionModel as CLIPVisionModelWithProjection;
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
    const inputs = this.tokenizer(text, {
      padding: true,
      truncation: true,
    });
    const { text_embeds } = await this.textModel(inputs);
    const embedding = text_embeds.data as Float32Array;
    return normalize(embedding);
  }

  async encodeImage(imageBlob: Blob): Promise<Float32Array> {
    if (!this.processor || !this.visionModel) {
      throw new Error("Vision model not loaded. Call loadVisionModel() first.");
    }
    const image = await RawImage.fromBlob(imageBlob);
    const inputs = await this.processor(image);
    const { image_embeds } = await this.visionModel(inputs);
    const embedding = image_embeds.data as Float32Array;
    return normalize(embedding);
  }
}
