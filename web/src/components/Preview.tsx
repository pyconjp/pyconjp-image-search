import { useCallback, useRef, useState } from "react";
import { flickrUrlResize } from "../lib/flickr";
import type { CropRect, FaceInfo, SearchResult } from "../types";
import { CropOverlay } from "./CropOverlay";
import { FaceThumbnails } from "./FaceThumbnails";
import { ThumbStrip } from "./ThumbStrip";

interface Props {
  results: SearchResult[];
  selectedIndex: number | null;
  faces: FaceInfo[];
  onSelect: (index: number) => void;
  onClose: () => void;
  onFindSimilar: (imageId: number) => void;
  onSearchCropped: (imageUrl: string, crop: CropRect) => void;
  onFindSamePerson: (faceIndex: number) => void;
}

export function Preview({
  results,
  selectedIndex,
  faces,
  onSelect,
  onClose,
  onFindSimilar,
  onSearchCropped,
  onFindSamePerson,
}: Props) {
  const imageRef = useRef<HTMLImageElement>(null);
  const [cropRect, setCropRect] = useState<CropRect | null>(null);

  const hasCrop = cropRect !== null;

  const handleCropChange = useCallback((crop: CropRect | null) => {
    setCropRect(crop);
  }, []);

  const selected = selectedIndex !== null ? results[selectedIndex] : null;
  if (!selected) return null;

  const previewUrl = flickrUrlResize(selected.image_url, "b");
  const flickrPageUrl = selected.flickr_photo_id
    ? `https://www.flickr.com/photos/pyconjp/${selected.flickr_photo_id}/`
    : null;

  const handleCopyToClipboard = async () => {
    if (!cropRect || !imageRef.current) return;
    try {
      const corsImg = new Image();
      corsImg.crossOrigin = "anonymous";
      await new Promise<void>((resolve, reject) => {
        corsImg.onload = () => resolve();
        corsImg.onerror = () => reject(new Error("Failed to load image"));
        corsImg.src = previewUrl;
      });
      const canvas = document.createElement("canvas");
      canvas.width = cropRect.w;
      canvas.height = cropRect.h;
      canvas
        .getContext("2d")
        ?.drawImage(
          corsImg,
          cropRect.x,
          cropRect.y,
          cropRect.w,
          cropRect.h,
          0,
          0,
          cropRect.w,
          cropRect.h,
        );
      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob(
          (b) => (b ? resolve(b) : reject(new Error("toBlob failed"))),
          "image/png",
        ),
      );
      await navigator.clipboard.write([
        new ClipboardItem({ "image/png": blob }),
      ]);
      showToast("Cropped area copied!", true);
    } catch {
      showToast("Failed to copy to clipboard", false);
    }
  };

  const handleSearchCropped = () => {
    if (!cropRect) return;
    onSearchCropped(previewUrl, cropRect);
  };

  return (
    <div className="preview-section">
      <div className="preview-image-wrapper">
        <img
          ref={imageRef}
          src={previewUrl}
          alt={selected.event_name}
          className="preview-image"
        />
        <CropOverlay imageRef={imageRef} onCropChange={handleCropChange} />
      </div>

      <div className="preview-caption">
        <span>
          score: {selected.score.toFixed(3)} | {selected.event_name}
        </span>
        {flickrPageUrl && (
          <a href={flickrPageUrl} target="_blank" rel="noopener noreferrer">
            Flickr
          </a>
        )}
      </div>

      {faces.length > 0 && (
        <FaceThumbnails
          imageUrl={previewUrl}
          faces={faces}
          onFaceClick={onFindSamePerson}
        />
      )}

      <div className="preview-actions">
        <button type="button" onClick={() => onFindSimilar(selected.id)}>
          Find Similar
        </button>
        <button type="button" onClick={handleSearchCropped} disabled={!hasCrop}>
          Search Cropped
        </button>
        <button
          type="button"
          onClick={handleCopyToClipboard}
          disabled={!hasCrop}
        >
          Copy to Clipboard
        </button>
        <button type="button" onClick={onClose}>
          Close
        </button>
      </div>

      <ThumbStrip
        results={results}
        selectedIndex={selectedIndex}
        onSelect={onSelect}
      />
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
