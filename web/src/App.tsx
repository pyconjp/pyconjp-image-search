import React, { useCallback, useEffect, useState } from "react";
import { EventFilter } from "./components/EventFilter";
import { Gallery } from "./components/Gallery";
import { ImageUpload } from "./components/ImageUpload";
import { LoadingOverlay } from "./components/LoadingOverlay";
import { LoadMoreButton } from "./components/LoadMoreButton";
import { LoginScreen } from "./components/LoginScreen";
import { Preview } from "./components/Preview";
import { SearchBar } from "./components/SearchBar";
import { TagFilter } from "./components/TagFilter";
import { AUTH_REQUIRED, useAuth } from "./hooks/useAuth";
import { DATASOURCE_TYPE, useDataSource } from "./hooks/useDataSource";
import { useDuckDB } from "./hooks/useDuckDB";
import { useEncoder } from "./hooks/useEncoder";
import { useImageSearch } from "./hooks/useImageSearch";
import { flickrUrlResize } from "./lib/flickr";
import { DEFAULT_CONFIG } from "./lib/models";
import type { CropRect, FaceInfo, SearchResult } from "./types";
import "./App.css";

type SearchMode = "text" | "image";

function revokeIfBlobUrl(url: string | null) {
  if (url?.startsWith("blob:")) URL.revokeObjectURL(url);
}

export default function App() {
  const config = DEFAULT_CONFIG;
  const {
    user,
    loading: authLoading,
    error: authError,
    signIn,
    signOut,
  } = useAuth();

  // ── Core hooks ───────────────────────────────────────────
  // Only load DuckDB when using duckdb datasource
  const needDuckDB = DATASOURCE_TYPE !== "firestore";
  const {
    conn,
    isLoading: dbLoading,
    error: dbError,
  } = useDuckDB(needDuckDB ? config.dbFileName : null);

  const dataSource = useDataSource(conn, config);

  const {
    encoder,
    isTextReady,
    isLoading: modelLoading,
    progress: modelProgress,
    error: modelError,
    loadVisionModel,
  } = useEncoder(config);
  const search = useImageSearch(dataSource, encoder);

  const [eventNames, setEventNames] = useState<string[]>([]);
  const [tagNames, setTagNames] = useState<string[]>([]);
  const [searchMode, setSearchMode] = useState<SearchMode>("text");
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const previewAnchorRef = React.useRef<HTMLDivElement>(null);
  const [sourceImageUrl, setSourceImageUrl] = useState<string | null>(null);
  const [faces, setFaces] = useState<FaceInfo[]>([]);
  const [activeFaceEmbeddings, setActiveFaceEmbeddings] = useState<
    number[][] | null
  >(null);

  // Load event names and tag names once data source is ready
  useEffect(() => {
    if (!dataSource) return;
    dataSource.getEventNames().then(setEventNames).catch(console.error);
    dataSource.getTagNames().then(setTagNames).catch(console.error);
  }, [dataSource]);

  // Fetch face detections when an image is selected
  useEffect(() => {
    if (!dataSource || selectedIndex === null) {
      setFaces([]);
      return;
    }
    const selected = search.results[selectedIndex];
    if (!selected) {
      setFaces([]);
      return;
    }
    let cancelled = false;
    dataSource
      .getFacesForImage(selected.id, selected.flickr_photo_id ?? undefined)
      .then((f) => {
        if (!cancelled) setFaces(f);
      })
      .catch(() => {
        if (!cancelled) setFaces([]);
      });
    return () => {
      cancelled = true;
    };
  }, [dataSource, selectedIndex, search.results]);

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

  const handleTextFullScan = useCallback(
    (query: string) => {
      const ok = window.confirm(
        "全件スキャンはFirestoreのコストが高くなります。テスト目的で数回のみ使用してください。",
      );
      if (!ok) return;
      setSelectedIndex(null);
      setActiveFaceEmbeddings(null);
      setSourceImageUrl((prev) => {
        revokeIfBlobUrl(prev);
        return null;
      });
      search.searchByText(query, search.selectedEvents, true);
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
    setTimeout(
      () => previewAnchorRef.current?.scrollIntoView({ behavior: "smooth" }),
      100,
    );
  }, []);

  const handleClosePreview = useCallback(() => {
    setSelectedIndex(null);
  }, []);

  const handleFindSimilar = useCallback(
    (result: SearchResult) => {
      setSourceImageUrl((prev) => {
        revokeIfBlobUrl(prev);
        return null;
      });
      setSourceImageUrl(flickrUrlResize(result.image_url, "z"));
      setSearchMode("image");
      setSelectedIndex(null);
      setActiveFaceEmbeddings(null);
      search.searchByStoredEmbedding(
        result.id,
        search.selectedEvents,
        result.flickr_photo_id ?? undefined,
      );
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
      setTimeout(
        () => previewAnchorRef.current?.scrollIntoView({ behavior: "smooth" }),
        100,
      );
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
      setTimeout(
        () => previewAnchorRef.current?.scrollIntoView({ behavior: "smooth" }),
        100,
      );
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

  const handleFullScanAsImage = useCallback(async () => {
    if (!sourceImageUrl) return;
    const ok = window.confirm(
      "全件スキャンはFirestoreのコストが高くなります。テスト目的で数回のみ使用してください。",
    );
    if (!ok) return;
    try {
      const resp = await fetch(sourceImageUrl);
      const blob = await resp.blob();
      setActiveFaceEmbeddings(null);
      await loadVisionModel();
      search.searchByImage(blob, search.selectedEvents, true);
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

  const handleTagsChange = useCallback(
    (tags: string[]) => {
      search.setSelectedTags(tags);
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

  // ── Auth gate ──────────────────────────────────────────
  if (AUTH_REQUIRED && authLoading) {
    return (
      <div className="loading-overlay">
        <div className="loading-content">
          <h2>PyCon JP Image Search</h2>
          <p>認証中...</p>
        </div>
      </div>
    );
  }

  if (AUTH_REQUIRED && !user) {
    return <LoginScreen onSignIn={signIn} error={authError} />;
  }

  // ── Loading screen ───────────────────────────────────────
  if ((needDuckDB && dbLoading) || modelLoading) {
    return (
      <LoadingOverlay
        dbReady={!dbLoading}
        modelReady={isTextReady}
        modelProgress={modelProgress}
        modelLabel={config.label}
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
        modelLabel={config.label}
        error={dbError || modelError}
      />
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>PyCon JP Image Search</h1>
        {AUTH_REQUIRED && user && (
          <button
            type="button"
            onClick={signOut}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              fontSize: "0.8rem",
              padding: "0.25rem 0.5rem",
              border: "1px solid #ccc",
              borderRadius: "4px",
              background: "transparent",
              cursor: "pointer",
            }}
          >
            {user.photoURL ? (
              <img
                src={user.photoURL}
                alt=""
                style={{
                  width: "1.4rem",
                  height: "1.4rem",
                  borderRadius: "50%",
                }}
                referrerPolicy="no-referrer"
              />
            ) : (
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: "1.4rem",
                  height: "1.4rem",
                  borderRadius: "50%",
                  background: "#666",
                  color: "#fff",
                  fontSize: "0.7rem",
                  fontWeight: "bold",
                }}
              >
                {(user.displayName || user.email || "?")
                  .charAt(0)
                  .toUpperCase()}
              </span>
            )}
            ログアウト
          </button>
        )}
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
            onFullScan={handleTextFullScan}
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
            onFullScanAsImage={handleFullScanAsImage}
            onReSearchByFaces={handleReSearchByFaces}
            onFullScan={() => {
              if (!activeFaceEmbeddings) return;
              const ok = window.confirm(
                "全件スキャンはFirestoreのコストが高くなります。テスト目的で数回のみ使用してください。",
              );
              if (!ok) return;
              search.searchByFaces(
                activeFaceEmbeddings,
                search.selectedEvents,
                true,
              );
            }}
          />
        )}

        <EventFilter
          eventNames={eventNames}
          selectedEvents={search.selectedEvents}
          onChange={handleEventsChange}
        />
        <TagFilter
          tags={tagNames}
          selectedTags={search.selectedTags}
          onChange={handleTagsChange}
        />
      </div>

      {search.error && (
        <div className="error-banner">
          <span>{search.error}</span>
          <button type="button" onClick={search.clearError}>
            x
          </button>
        </div>
      )}

      {search.message && <p className="search-message">{search.message}</p>}

      <div ref={previewAnchorRef} />
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

      {search.isSearching && (
        <div className="search-loading">
          <div className="search-spinner" />
          <span>検索中...</span>
        </div>
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
