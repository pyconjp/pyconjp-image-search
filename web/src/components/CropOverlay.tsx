import { useCallback, useEffect, useRef, useState } from "react";
import type { CropRect } from "../types";

interface Props {
  imageRef: React.RefObject<HTMLImageElement | null>;
  onCropChange: (crop: CropRect | null) => void;
}

/** Compute the actual rendered image rect inside an object-fit:contain element. */
function getRenderedRect(img: HTMLImageElement) {
  const ir = img.getBoundingClientRect();
  const natW = img.naturalWidth;
  const natH = img.naturalHeight;
  if (!natW || !natH) return ir;
  const style = window.getComputedStyle(img);
  const fit = style.objectFit || "fill";
  if (fit === "contain" || fit === "scale-down") {
    const ratio = Math.min(ir.width / natW, ir.height / natH);
    const rw = natW * ratio;
    const rh = natH * ratio;
    return {
      left: ir.left + (ir.width - rw) / 2,
      top: ir.top + (ir.height - rh) / 2,
      width: rw,
      height: rh,
    };
  }
  return { left: ir.left, top: ir.top, width: ir.width, height: ir.height };
}

export function CropOverlay({ imageRef, onCropChange }: Props) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const selRef = useRef<HTMLDivElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const startRef = useRef({ x: 0, y: 0 });
  const currentRef = useRef({ x: 0, y: 0 });

  const updateSelectionDiv = useCallback(() => {
    const sel = selRef.current;
    const img = imageRef.current;
    if (!sel || !img) return;

    const wrapper = img.parentElement;
    if (!wrapper) return;

    const wr = wrapper.getBoundingClientRect();
    const rr = getRenderedRect(img);
    const offX = rr.left - wr.left;
    const offY = rr.top - wr.top;

    const s = startRef.current;
    const e = currentRef.current;
    const x = Math.min(s.x, e.x) + offX;
    const y = Math.min(s.y, e.y) + offY;
    const w = Math.abs(e.x - s.x);
    const h = Math.abs(e.y - s.y);

    sel.style.left = `${x}px`;
    sel.style.top = `${y}px`;
    sel.style.width = `${w}px`;
    sel.style.height = `${h}px`;
  }, [imageRef]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      const img = imageRef.current;
      if (!img) return;

      const rr = getRenderedRect(img);
      const x = e.clientX - rr.left;
      const y = e.clientY - rr.top;
      if (x < 0 || y < 0 || x > rr.width || y > rr.height) return;

      startRef.current = { x, y };
      currentRef.current = { x, y };
      setIsDrawing(true);
      onCropChange(null);

      if (selRef.current) {
        selRef.current.style.display = "block";
      }
      updateSelectionDiv();
    },
    [imageRef, onCropChange, updateSelectionDiv],
  );

  useEffect(() => {
    if (!isDrawing) return;

    const img = imageRef.current;
    if (!img) return;

    const handleMouseMove = (e: MouseEvent) => {
      const rr = getRenderedRect(img);
      currentRef.current = {
        x: Math.max(0, Math.min(e.clientX - rr.left, rr.width)),
        y: Math.max(0, Math.min(e.clientY - rr.top, rr.height)),
      };
      updateSelectionDiv();
    };

    const handleMouseUp = () => {
      setIsDrawing(false);
      const s = startRef.current;
      const c = currentRef.current;
      const w = Math.abs(c.x - s.x);
      const h = Math.abs(c.y - s.y);

      if (w > 5 && h > 5) {
        const rr = getRenderedRect(img);
        const scX = img.naturalWidth / rr.width;
        const scY = img.naturalHeight / rr.height;
        onCropChange({
          x: Math.round(Math.min(s.x, c.x) * scX),
          y: Math.round(Math.min(s.y, c.y) * scY),
          w: Math.round(w * scX),
          h: Math.round(h * scY),
        });
      } else {
        if (selRef.current) selRef.current.style.display = "none";
        onCropChange(null);
      }
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDrawing, imageRef, onCropChange, updateSelectionDiv]);

  return (
    <>
      <div
        ref={overlayRef}
        className="crop-overlay"
        onMouseDown={handleMouseDown}
      />
      <div
        ref={selRef}
        className="crop-selection"
        style={{ display: "none" }}
      />
    </>
  );
}
