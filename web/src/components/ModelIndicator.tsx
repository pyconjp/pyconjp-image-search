interface Props {
  modelLabel: string;
  onChangeModel: () => void;
  onClearCache: () => void;
}

export function ModelIndicator({
  modelLabel,
  onChangeModel,
  onClearCache,
}: Props) {
  return (
    <div className="model-indicator">
      <span className="model-indicator-label">Model: {modelLabel}</span>
      <button
        type="button"
        className="model-indicator-btn"
        onClick={onChangeModel}
      >
        Change
      </button>
      <button
        type="button"
        className="model-indicator-btn"
        onClick={onClearCache}
      >
        Clear Cache
      </button>
    </div>
  );
}
