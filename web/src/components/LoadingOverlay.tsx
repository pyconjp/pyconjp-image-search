interface Props {
  dbReady: boolean;
  modelReady: boolean;
  modelProgress: number;
  modelLabel: string;
  error: string | null;
}

export function LoadingOverlay({
  dbReady,
  modelReady,
  modelProgress,
  modelLabel,
  error,
}: Props) {
  if (error) {
    return (
      <div className="loading-overlay">
        <div className="loading-content">
          <h2>Error</h2>
          <p className="error-message">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="loading-overlay">
      <div className="loading-content">
        <h2>PyCon JP Image Search</h2>
        <p>Initializing {modelLabel}...</p>
        <div className="loading-steps">
          <div className={`loading-step ${dbReady ? "done" : "active"}`}>
            {dbReady ? "\u2713" : "\u25cb"} Loading database...
          </div>
          <div
            className={`loading-step ${modelReady ? "done" : dbReady ? "active" : ""}`}
          >
            {modelReady ? "\u2713" : "\u25cb"} Loading {modelLabel} model...
            {dbReady && !modelReady && modelProgress > 0 && (
              <span className="progress-text">
                {" "}
                {Math.round(modelProgress)}%
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
