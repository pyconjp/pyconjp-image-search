interface LoginScreenProps {
  onSignIn: () => Promise<void>;
  error?: string | null;
}

export function LoginScreen({ onSignIn, error }: LoginScreenProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        gap: "1.5rem",
        padding: "2rem",
      }}
    >
      <h1>PyCon JP Image Search</h1>
      <p style={{ color: "#666", textAlign: "center" }}>
        @pycon.jp の Google アカウントでログインしてください
      </p>
      <button
        type="button"
        onClick={onSignIn}
        style={{
          padding: "0.75rem 2rem",
          fontSize: "1rem",
          borderRadius: "8px",
          border: "1px solid #ccc",
          background: "#fff",
          cursor: "pointer",
        }}
      >
        Google でログイン
      </button>
      {error && (
        <p style={{ color: "red", textAlign: "center", maxWidth: "400px" }}>
          {error}
        </p>
      )}
    </div>
  );
}
