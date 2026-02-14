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
  const [activeFaceEmbedding, setActiveFaceEmbedding] = useState<
    number[] | null
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
      setActiveFaceEmbedding(null);
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
      setActiveFaceEmbedding(null);
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
      setActiveFaceEmbedding(null);
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
      setActiveFaceEmbedding(null);
      search.searchByImage(blob, search.selectedEvents);
    },
    [search, loadVisionModel],
  );

  const handleFindSamePerson = useCallback(
    async (faceIndex: number) => {
      if (faceIndex >= faces.length || selectedIndex === null) return;
      const face = faces[faceIndex];
      const selected = search.results[selectedIndex];
      if (!selected) return;

      // Create face crop from the preview image
      const previewUrl = flickrUrlResize(selected.image_url, "b");
      try {
        const corsImg = new Image();
        corsImg.crossOrigin = "anonymous";
        await new Promise<void>((resolve, reject) => {
          corsImg.onload = () => resolve();
          corsImg.onerror = () => reject(new Error("Failed to load image"));
          corsImg.src = previewUrl;
        });
        const scaleX = corsImg.naturalWidth / face.image_width;
        const scaleY = corsImg.naturalHeight / face.image_height;
        const [x1, y1, x2, y2] = face.bbox;
        const fw = Math.round(x2 * scaleX) - Math.round(x1 * scaleX);
        const fh = Math.round(y2 * scaleY) - Math.round(y1 * scaleY);
        const padX = Math.round(fw * 0.1);
        const padY = Math.round(fh * 0.1);
        const cx1 = Math.max(0, Math.round(x1 * scaleX) - padX);
        const cy1 = Math.max(0, Math.round(y1 * scaleY) - padY);
        const cx2 = Math.min(
          corsImg.naturalWidth,
          Math.round(x2 * scaleX) + padX,
        );
        const cy2 = Math.min(
          corsImg.naturalHeight,
          Math.round(y2 * scaleY) + padY,
        );
        const cw = cx2 - cx1;
        const ch = cy2 - cy1;
        const canvas = document.createElement("canvas");
        canvas.width = cw;
        canvas.height = ch;
        canvas
          .getContext("2d")
          ?.drawImage(corsImg, cx1, cy1, cw, ch, 0, 0, cw, ch);
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

      setActiveFaceEmbedding(face.embedding);
      setSelectedIndex(null);
      setSearchMode("image");
      search.searchByFace(face.embedding, search.selectedEvents);
    },
    [faces, search, selectedIndex],
  );

  const handleSearchFaceAsImage = useCallback(async () => {
    if (!sourceImageUrl) return;
    try {
      const resp = await fetch(sourceImageUrl);
      const blob = await resp.blob();
      setActiveFaceEmbedding(null);
      await loadVisionModel();
      search.searchByImage(blob, search.selectedEvents);
    } catch {
      // ignore
    }
  }, [sourceImageUrl, search, loadVisionModel]);

  const handleReSearchByFace = useCallback(() => {
    if (!activeFaceEmbedding) return;
    search.searchByFace(activeFaceEmbedding, search.selectedEvents);
  }, [activeFaceEmbedding, search]);

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
            activeFaceEmbedding={activeFaceEmbedding}
            onSearchAsImage={handleSearchFaceAsImage}
            onReSearchByFace={handleReSearchByFace}
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
          onSelect={setSelectedIndex}
          onClose={handleClosePreview}
          onFindSimilar={handleFindSimilar}
          onSearchCropped={handleSearchCropped}
          onFindSamePerson={handleFindSamePerson}
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
