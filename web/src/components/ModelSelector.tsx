import { MODEL_CONFIGS } from "../lib/models";

interface Props {
  onSelect: (modelId: string) => void;
}

export function ModelSelector({ onSelect }: Props) {
  return (
    <div className="model-selector-overlay">
      <div className="model-selector-content">
        <h2>PyCon JP Image Search</h2>
        <p>Select an embedding model to get started.</p>
        <div className="model-cards">
          {MODEL_CONFIGS.map((config) => (
            <button
              key={config.id}
              type="button"
              className="model-card"
              onClick={() => onSelect(config.id)}
            >
              <h3>{config.label}</h3>
              <div className="model-card-details">
                <span>Dim: {config.embeddingDim}</span>
                <span>{config.useFp16 ? "FP16" : "FP32"}</span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
