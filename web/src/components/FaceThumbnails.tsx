import { useEffect, useRef, useState } from "react";
import type { FaceInfo } from "../types";

interface Props {
  imageUrl: string;
  faces: FaceInfo[];
  selectedIndices: number[];
  onToggleFace: (faceIndex: number) => void;
}

export function FaceThumbnails({
  imageUrl,
  faces,
  selectedIndices,
  onToggleFace,
}: Props) {
  const [cropUrls, setCropUrls] = useState<string[]>([]);
  const prevUrlRef = useRef<string>("");

  useEffect(() => {
    if (!imageUrl || faces.length === 0) {
      setCropUrls([]);
      return;
    }

    // Avoid re-cropping if same image
    if (prevUrlRef.current === imageUrl && cropUrls.length === faces.length) {
      return;
    }
    prevUrlRef.current = imageUrl;

    let cancelled = false;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (cancelled) return;

      // Scale factor: bbox coords are in original image pixels,
      // but the loaded image (_b size) may differ
      const face0 = faces[0]!;
      const scaleX = img.naturalWidth / face0.image_width;
      const scaleY = img.naturalHeight / face0.image_height;

      const urls: string[] = [];
      for (const face of faces) {
        const [x1, y1, x2, y2] = face.bbox;
        const fx1 = Math.round(x1 * scaleX);
        const fy1 = Math.round(y1 * scaleY);
        const fx2 = Math.round(x2 * scaleX);
        const fy2 = Math.round(y2 * scaleY);

        // Add 10% padding
        const fw = fx2 - fx1;
        const fh = fy2 - fy1;
        const padX = Math.round(fw * 0.1);
        const padY = Math.round(fh * 0.1);
        const cx1 = Math.max(0, fx1 - padX);
        const cy1 = Math.max(0, fy1 - padY);
        const cx2 = Math.min(img.naturalWidth, fx2 + padX);
        const cy2 = Math.min(img.naturalHeight, fy2 + padY);
        const cw = cx2 - cx1;
        const ch = cy2 - cy1;

        const canvas = document.createElement("canvas");
        canvas.width = cw;
        canvas.height = ch;
        canvas.getContext("2d")?.drawImage(img, cx1, cy1, cw, ch, 0, 0, cw, ch);
        urls.push(canvas.toDataURL("image/jpeg", 0.85));
      }

      if (!cancelled) {
        setCropUrls(urls);
      }
    };
    img.onerror = () => {
      if (!cancelled) setCropUrls([]);
    };
    img.src = imageUrl;

    return () => {
      cancelled = true;
    };
  }, [imageUrl, faces, cropUrls.length]);

  if (faces.length === 0 || cropUrls.length !== faces.length) return null;

  return (
    <div className="face-thumbnails">
      <span className="face-thumbnails-label">Faces:</span>
      {cropUrls.map((url, i) => {
        const face = faces[i]!;
        return (
          <button
            type="button"
            key={face.face_id}
            className={`face-thumb${selectedIndices.includes(i) ? " selected" : ""}`}
            onClick={() => onToggleFace(i)}
            title={`Select face ${i + 1} (score: ${face.det_score.toFixed(2)})`}
          >
            <img src={url} alt={`Face ${i + 1}`} />
          </button>
        );
      })}
    </div>
  );
}
