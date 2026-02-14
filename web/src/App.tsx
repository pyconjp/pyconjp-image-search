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
import { getEventNames } from "./lib/search";
import type { CropRect } from "./types";
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

  // Load event names once DB is ready
  useEffect(() => {
    if (!conn) return;
    getEventNames(conn).then(setEventNames).catch(console.error);
  }, [conn]);

  const handleTextSearch = useCallback(
    (query: string) => {
      setSelectedIndex(null);
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
      search.searchByImage(blob, search.selectedEvents);
    },
    [search, loadVisionModel],
  );

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
          onSelect={setSelectedIndex}
          onClose={handleClosePreview}
          onFindSimilar={handleFindSimilar}
          onSearchCropped={handleSearchCropped}
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
