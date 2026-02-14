import { type ChangeEvent, useRef } from "react";

interface Props {
  onUpload: (blob: Blob) => void;
  isSearching: boolean;
  disabled: boolean;
  sourceImageUrl: string | null;
}

export function ImageUpload({
  onUpload,
  isSearching,
  disabled,
  sourceImageUrl,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUpload(file);
    }
  };

  const handlePasteFromClipboard = async () => {
    try {
      const items = await navigator.clipboard.read();
      for (const item of items) {
        const imageType = item.types.find((t) => t.startsWith("image/"));
        if (imageType) {
          const blob = await item.getType(imageType);
          onUpload(blob);
          return;
        }
      }
      showToast("No image found in clipboard", false);
    } catch {
      showToast("Could not read clipboard. Try Ctrl+V instead.", false);
    }
  };

  return (
    <div className="image-upload">
      <div className="image-upload-actions">
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          onChange={handleChange}
          disabled={disabled || isSearching}
          className="image-upload-input"
        />
        <button
          type="button"
          className="image-upload-button"
          disabled={disabled || isSearching}
          onClick={() => fileRef.current?.click()}
        >
          {isSearching ? "Searching..." : "Upload Image"}
        </button>
        <button
          type="button"
          className="image-paste-button"
          disabled={disabled || isSearching}
          onClick={handlePasteFromClipboard}
        >
          Paste from Clipboard
        </button>
        <span className="image-upload-hint">or Ctrl+V</span>
      </div>
      {sourceImageUrl && (
        <div className="source-image-preview">
          <span className="source-image-label">Query image:</span>
          <img src={sourceImageUrl} alt="Search source" />
        </div>
      )}
    </div>
  );
}

function showToast(msg: string, ok: boolean) {
  const t = document.createElement("div");
  t.textContent = msg;
  t.style.cssText = `position:fixed;top:20px;left:50%;transform:translateX(-50%);
    background:${ok ? "#333" : "#c00"};color:#fff;padding:12px 24px;
    border-radius:8px;z-index:9999;font-size:14px;box-shadow:0 2px 8px rgba(0,0,0,.3);`;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2000);
}
