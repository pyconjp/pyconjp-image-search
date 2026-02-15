import { useCallback, useEffect, useState } from "react";
import { EventFilter } from "./components/EventFilter";
import { Gallery } from "./components/Gallery";
import { ImageUpload } from "./components/ImageUpload";
import { LoadingOverlay } from "./components/LoadingOverlay";
import { LoadMoreButton } from "./components/LoadMoreButton";
import { Preview } from "./components/Preview";
import { SearchBar } from "./components/SearchBar";
import { useCLIPEncoder } from "./hooks/useCLIPEncoder";
import { useDuckDB } from "./hooks/useDuckDB";
import { useImageSearch } from "./hooks/useImageSearch";
import { flickrUrlResize } from "./lib/flickr";
import { getEventNames, getFacesForImage } from "./lib/search";
import type { CropRect, FaceInfo } from "./types";
import "./App.css";

type SearchMode = "text" | "image";

function revokeIfBlobUrl(url: string | null) {
  if (url?.startsWith("blob:")) URL.revokeObjectURL(url);
}

export default function App() {
  const { conn, isLoading: dbLoading, error: dbError } = useDuckDB();
  const {
    encoder,
    isTextReady,
    isLoading: modelLoading,
    progress: modelProgress,
    error: modelError,
    loadVisionModel,
  } = useCLIPEncoder();
  const search = useImageSearch(conn, encoder);

  const [eventNames, setEventNames] = useState<string[]>([]);
  const [searchMode, setSearchMode] = useState<SearchMode>("text");
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [sourceImageUrl, setSourceImageUrl] = useState<string | null>(null);
  const [faces, setFaces] = useState<FaceInfo[]>([]);
  const [activeFaceEmbeddings, setActiveFaceEmbeddings] = useState<
    number[][] | null
  >(null);

  // Load event names once DB is ready
  useEffect(() => {
    if (!conn) return;
    getEventNames(conn).then(setEventNames).catch(console.error);
  }, [conn]);

  // Fetch face detections when an image is selected
  useEffect(() => {
    if (!conn || selectedIndex === null) {
      setFaces([]);
      return;
    }
    const selected = search.results[selectedIndex];
    if (!selected) {
      setFaces([]);
      return;
    }
    let cancelled = false;
    getFacesForImage(conn, selected.id)
      .then((f) => {
        if (!cancelled) setFaces(f);
      })
      .catch(() => {
        if (!cancelled) setFaces([]);
      });
    return () => {
      cancelled = true;
    };
  }, [conn, selectedIndex, search.results]);

  const handleTextSearch = useCallback(
    (query: string) => {
      setSelectedIndex(null);
      setActiveFaceEmbeddings(null);
      setSourceImageUrl((prev) => {
        revokeIfBlobUrl(prev);
        return null;
      });
      search.searchByText(query, search.selectedEvents);
    },
    [search],
  );

  const handleImageUpload = useCallback(
    async (blob: Blob) => {
      setSelectedIndex(null);
      setActiveFaceEmbeddings(null);
      // Show preview of the source image
      const url = URL.createObjectURL(blob);
      setSourceImageUrl((prev) => {
        revokeIfBlobUrl(prev);
        return url;
      });
      // Load vision model on first image upload
      await loadVisionModel();
      search.searchByImage(blob, search.selectedEvents);
    },
    [search, loadVisionModel],
  );

  const handleGallerySelect = useCallback((index: number) => {
    setSelectedIndex(index);
  }, []);

  const handleClosePreview = useCallback(() => {
    setSelectedIndex(null);
  }, []);

  const handleFindSimilar = useCallback(
    (imageId: number) => {
      // Show the selected image as source preview
      const match = search.results.find((r) => r.id === imageId);
      if (match) {
        setSourceImageUrl((prev) => {
          revokeIfBlobUrl(prev);
          return null;
        });
        setSourceImageUrl(flickrUrlResize(match.image_url, "z"));
        setSearchMode("image");
      }
      setSelectedIndex(null);
      setActiveFaceEmbeddings(null);
      search.searchByStoredEmbedding(imageId, search.selectedEvents);
    },
    [search],
  );

  const handleSearchCropped = useCallback(
    async (imageUrl: string, crop: CropRect) => {
      // Load vision model if not ready
      await loadVisionModel();

      // Fetch the image via canvas crop
      const corsImg = new Image();
      corsImg.crossOrigin = "anonymous";
      await new Promise<void>((resolve, reject) => {
        corsImg.onload = () => resolve();
        corsImg.onerror = () => reject(new Error("Failed to load image"));
        corsImg.src = imageUrl;
      });
      const canvas = document.createElement("canvas");
      canvas.width = crop.w;
      canvas.height = crop.h;
      canvas
        .getContext("2d")
        ?.drawImage(
          corsImg,
          crop.x,
          crop.y,
          crop.w,
          crop.h,
          0,
          0,
          crop.w,
          crop.h,
        );
      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob(
          (b) => (b ? resolve(b) : reject(new Error("toBlob failed"))),
          "image/png",
        ),
      );

      // Show cropped image as source
      const url = URL.createObjectURL(blob);
      setSourceImageUrl((prev) => {
        revokeIfBlobUrl(prev);
        return url;
      });

      setSelectedIndex(null);
      setActiveFaceEmbeddings(null);
      search.searchByImage(blob, search.selectedEvents);
    },
    [search, loadVisionModel],
  );

  const handleFindSamePersons = useCallback(
    async (faceIndices: number[]) => {
      if (faceIndices.length === 0 || selectedIndex === null) return;
      const selectedFaces = faceIndices
        .map((i) => faces[i])
        .filter((f): f is FaceInfo => f != null);
      if (selectedFaces.length === 0) return;
      const selected = search.results[selectedIndex];
      if (!selected) return;

      // Create composite face crop of all selected faces for source image
      const previewUrl = flickrUrlResize(selected.image_url, "b");
      try {
        const corsImg = new Image();
        corsImg.crossOrigin = "anonymous";
        await new Promise<void>((resolve, reject) => {
          corsImg.onload = () => resolve();
          corsImg.onerror = () => reject(new Error("Failed to load image"));
          corsImg.src = previewUrl;
        });
        // Compute crop regions for all selected faces
        const crops: { x: number; y: number; w: number; h: number }[] = [];
        for (const face of selectedFaces) {
          const sx = corsImg.naturalWidth / face.image_width;
          const sy = corsImg.naturalHeight / face.image_height;
          const [x1, y1, x2, y2] = face.bbox;
          const fw = Math.round(x2 * sx) - Math.round(x1 * sx);
          const fh = Math.round(y2 * sy) - Math.round(y1 * sy);
          const padX = Math.round(fw * 0.1);
          const padY = Math.round(fh * 0.1);
          crops.push({
            x: Math.max(0, Math.round(x1 * sx) - padX),
            y: Math.max(0, Math.round(y1 * sy) - padY),
            w:
              Math.min(corsImg.naturalWidth, Math.round(x2 * sx) + padX) -
              Math.max(0, Math.round(x1 * sx) - padX),
            h:
              Math.min(corsImg.naturalHeight, Math.round(y2 * sy) + padY) -
              Math.max(0, Math.round(y1 * sy) - padY),
          });
        }
        // Composite: arrange all face crops side by side, normalized to same height
        const targetH = Math.max(...crops.map((c) => c.h));
        const gap = 4;
        const totalW =
          crops.reduce((s, c) => s + Math.round((c.w * targetH) / c.h), 0) +
          gap * (crops.length - 1);
        const canvas = document.createElement("canvas");
        canvas.width = totalW;
        canvas.height = targetH;
        const ctx = canvas.getContext("2d");
        let xOff = 0;
        for (const crop of crops) {
          const scaledW = Math.round((crop.w * targetH) / crop.h);
          ctx?.drawImage(
            corsImg,
            crop.x,
            crop.y,
            crop.w,
            crop.h,
            xOff,
            0,
            scaledW,
            targetH,
          );
          xOff += scaledW + gap;
        }
        const blob = await new Promise<Blob>((resolve, reject) =>
          canvas.toBlob(
            (b) => (b ? resolve(b) : reject(new Error("toBlob failed"))),
            "image/jpeg",
            0.85,
          ),
        );
        const url = URL.createObjectURL(blob);
        setSourceImageUrl((prev) => {
          revokeIfBlobUrl(prev);
          return url;
        });
      } catch {
        setSourceImageUrl((prev) => {
          revokeIfBlobUrl(prev);
          return null;
        });
      }

      const embeddings = selectedFaces.map((f) => f.embedding);
      setActiveFaceEmbeddings(embeddings);
      setSelectedIndex(null);
      setSearchMode("image");
      search.searchByFaces(embeddings, search.selectedEvents);
    },
    [faces, search, selectedIndex],
  );

  const handleAddFacesToQuery = useCallback(
    async (faceIndices: number[]) => {
      if (faceIndices.length === 0 || selectedIndex === null) return;
      const newFaces = faceIndices
        .map((i) => faces[i])
        .filter((f): f is FaceInfo => f != null);
      if (newFaces.length === 0) return;
      const selected = search.results[selectedIndex];
      if (!selected) return;

      // Merge new embeddings with existing ones
      const existingEmbeddings = activeFaceEmbeddings ?? [];
      const mergedEmbeddings = [
        ...existingEmbeddings,
        ...newFaces.map((f) => f.embedding),
      ];

      // Build new composite source image: existing source + new face crops
      try {
        const previewUrl = flickrUrlResize(selected.image_url, "b");
        const corsImg = new Image();
        corsImg.crossOrigin = "anonymous";
        await new Promise<void>((resolve, reject) => {
          corsImg.onload = () => resolve();
          corsImg.onerror = () => reject(new Error("Failed to load image"));
          corsImg.src = previewUrl;
        });

        // Crop new faces
        const newCropCanvases: HTMLCanvasElement[] = [];
        for (const face of newFaces) {
          const sx = corsImg.naturalWidth / face.image_width;
          const sy = corsImg.naturalHeight / face.image_height;
          const [x1, y1, x2, y2] = face.bbox;
          const fw = Math.round(x2 * sx) - Math.round(x1 * sx);
          const fh = Math.round(y2 * sy) - Math.round(y1 * sy);
          const padX = Math.round(fw * 0.1);
          const padY = Math.round(fh * 0.1);
          const cx = Math.max(0, Math.round(x1 * sx) - padX);
          const cy = Math.max(0, Math.round(y1 * sy) - padY);
          const cw =
            Math.min(corsImg.naturalWidth, Math.round(x2 * sx) + padX) - cx;
          const ch =
            Math.min(corsImg.naturalHeight, Math.round(y2 * sy) + padY) - cy;
          const c = document.createElement("canvas");
          c.width = cw;
          c.height = ch;
          c.getContext("2d")?.drawImage(corsImg, cx, cy, cw, ch, 0, 0, cw, ch);
          newCropCanvases.push(c);
        }

        // Load existing source image (if any)
        const parts: HTMLCanvasElement[] = [];
        if (sourceImageUrl) {
          const oldImg = new Image();
          await new Promise<void>((resolve, reject) => {
            oldImg.onload = () => resolve();
            oldImg.onerror = () => reject(new Error("Failed to load source"));
            oldImg.src = sourceImageUrl;
          });
          const c = document.createElement("canvas");
          c.width = oldImg.naturalWidth;
          c.height = oldImg.naturalHeight;
          c.getContext("2d")?.drawImage(oldImg, 0, 0);
          parts.push(c);
        }
        parts.push(...newCropCanvases);

        // Composite all parts side by side, normalized to same height
        const targetH = Math.max(...parts.map((c) => c.height));
        const gap = 4;
        const totalW =
          parts.reduce(
            (s, c) => s + Math.round((c.width * targetH) / c.height),
            0,
          ) +
          gap * (parts.length - 1);
        const canvas = document.createElement("canvas");
        canvas.width = totalW;
        canvas.height = targetH;
        const ctx = canvas.getContext("2d");
        let xOff = 0;
        for (const part of parts) {
          const scaledW = Math.round((part.width * targetH) / part.height);
          ctx?.drawImage(
            part,
            0,
            0,
            part.width,
            part.height,
            xOff,
            0,
            scaledW,
            targetH,
          );
          xOff += scaledW + gap;
        }
        const blob = await new Promise<Blob>((resolve, reject) =>
          canvas.toBlob(
            (b) => (b ? resolve(b) : reject(new Error("toBlob failed"))),
            "image/jpeg",
            0.85,
          ),
        );
        const url = URL.createObjectURL(blob);
        setSourceImageUrl((prev) => {
          revokeIfBlobUrl(prev);
          return url;
        });
      } catch {
        // Keep existing source image on error
      }

      setActiveFaceEmbeddings(mergedEmbeddings);
      setSelectedIndex(null);
      setSearchMode("image");
      search.searchByFaces(mergedEmbeddings, search.selectedEvents);
    },
    [faces, search, selectedIndex, activeFaceEmbeddings, sourceImageUrl],
  );

  const handleSearchFaceAsImage = useCallback(async () => {
    if (!sourceImageUrl) return;
    try {
      const resp = await fetch(sourceImageUrl);
      const blob = await resp.blob();
      setActiveFaceEmbeddings(null);
      await loadVisionModel();
      search.searchByImage(blob, search.selectedEvents);
    } catch {
      // ignore
    }
  }, [sourceImageUrl, search, loadVisionModel]);

  const handleReSearchByFaces = useCallback(() => {
    if (!activeFaceEmbeddings || activeFaceEmbeddings.length === 0) return;
    search.searchByFaces(activeFaceEmbeddings, search.selectedEvents);
  }, [activeFaceEmbeddings, search]);

  const handleEventsChange = useCallback(
    (events: string[]) => {
      search.setSelectedEvents(events);
    },
    [search],
  );

  // Global paste handler for clipboard image search
  useEffect(() => {
    const handlePaste = async (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          e.preventDefault();
          const blob = item.getAsFile();
          if (blob) {
            setSearchMode("image");
            await handleImageUpload(blob);
          }
          return;
        }
      }
    };
    document.addEventListener("paste", handlePaste);
    return () => document.removeEventListener("paste", handlePaste);
  }, [handleImageUpload]);

  // Show loading screen
  if (dbLoading || modelLoading) {
    return (
      <LoadingOverlay
        dbReady={!dbLoading}
        modelReady={isTextReady}
        modelProgress={modelProgress}
        error={dbError || modelError}
      />
    );
  }

  if (dbError || modelError) {
    return (
      <LoadingOverlay
        dbReady={!dbLoading}
        modelReady={isTextReady}
        modelProgress={0}
        error={dbError || modelError}
      />
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>PyCon JP Image Search</h1>
      </header>

      <div className="search-controls">
        <div className="mode-toggle">
          <button
            type="button"
            className={searchMode === "text" ? "active" : ""}
            onClick={() => setSearchMode("text")}
          >
            Text Search
          </button>
          <button
            type="button"
            className={searchMode === "image" ? "active" : ""}
            onClick={() => setSearchMode("image")}
          >
            Image Search
          </button>
        </div>

        {searchMode === "text" ? (
          <SearchBar
            onSearch={handleTextSearch}
            isSearching={search.isSearching}
            disabled={!isTextReady}
          />
        ) : (
          <ImageUpload
            onUpload={handleImageUpload}
            isSearching={search.isSearching}
            disabled={!isTextReady}
            sourceImageUrl={sourceImageUrl}
            activeFaceEmbeddings={activeFaceEmbeddings}
            onSearchAsImage={handleSearchFaceAsImage}
            onReSearchByFaces={handleReSearchByFaces}
          />
        )}

        <EventFilter
          eventNames={eventNames}
          selectedEvents={search.selectedEvents}
          onChange={handleEventsChange}
        />
      </div>

      {search.message && <p className="search-message">{search.message}</p>}

      {selectedIndex !== null && (
        <Preview
          results={search.results}
          selectedIndex={selectedIndex}
          faces={faces}
          hasActiveFaceQuery={
            activeFaceEmbeddings != null && activeFaceEmbeddings.length > 0
          }
          onSelect={setSelectedIndex}
          onClose={handleClosePreview}
          onFindSimilar={handleFindSimilar}
          onSearchCropped={handleSearchCropped}
          onFindSamePersons={handleFindSamePersons}
          onAddFacesToQuery={handleAddFacesToQuery}
        />
      )}

      <Gallery results={search.results} onSelect={handleGallerySelect} />

      <LoadMoreButton
        onClick={search.loadMore}
        isLoading={search.isSearching}
        visible={search.hasMore}
      />
    </div>
  );
}
