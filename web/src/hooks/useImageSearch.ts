import { useCallback, useState } from "react";
import type { DataSource } from "../lib/datasource";
import type { VisionLanguageEncoder } from "../lib/encoder";
import type { SearchResult } from "../types";

const PAGE_SIZE = 20;

export function useImageSearch(
  dataSource: DataSource | null,
  encoder: VisionLanguageEncoder | null,
) {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [currentEmbedding, setCurrentEmbedding] = useState<Float32Array | null>(
    null,
  );
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [message, setMessage] = useState("");
  const [selectedEvents, setSelectedEvents] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [currentFaceEmbeddings, setCurrentFaceEmbeddings] = useState<
    number[][] | null
  >(null);
  const [fullScan, setFullScan] = useState(false);

  const searchByText = useCallback(
    async (query: string, eventNames?: string[]) => {
      if (!dataSource || !encoder) return;
      if (!query.trim()) {
        setMessage("Please enter a search query.");
        return;
      }
      setIsSearching(true);
      try {
        const embedding = await encoder.encodeText(query);
        const events = eventNames ?? selectedEvents;
        const hits = await dataSource.searchByEmbedding(embedding, {
          limit: PAGE_SIZE,
          offset: 0,
          eventNames: events.length > 0 ? events : undefined,
          tagNames: selectedTags.length > 0 ? selectedTags : undefined,
        });
        setResults(hits);
        setCurrentEmbedding(embedding);
        setCurrentFaceEmbeddings(null);
        setOffset(PAGE_SIZE);
        setHasMore(hits.length === PAGE_SIZE);
        setMessage(`Found ${hits.length} images for "${query}".`);
        if (eventNames) setSelectedEvents(eventNames);
      } finally {
        setIsSearching(false);
      }
    },
    [dataSource, encoder, selectedEvents, selectedTags],
  );

  const searchByImage = useCallback(
    async (imageBlob: Blob, eventNames?: string[]) => {
      if (!dataSource || !encoder) return;
      setIsSearching(true);
      try {
        const embedding = await encoder.encodeImage(imageBlob);
        const events = eventNames ?? selectedEvents;
        const hits = await dataSource.searchByEmbedding(embedding, {
          limit: PAGE_SIZE,
          offset: 0,
          eventNames: events.length > 0 ? events : undefined,
          tagNames: selectedTags.length > 0 ? selectedTags : undefined,
        });
        setResults(hits);
        setCurrentEmbedding(embedding);
        setCurrentFaceEmbeddings(null);
        setOffset(PAGE_SIZE);
        setHasMore(hits.length === PAGE_SIZE);
        setMessage(`Found ${hits.length} similar images.`);
        if (eventNames) setSelectedEvents(eventNames);
      } finally {
        setIsSearching(false);
      }
    },
    [dataSource, encoder, selectedEvents, selectedTags],
  );

  const searchByStoredEmbedding = useCallback(
    async (imageId: number, eventNames?: string[], flickrPhotoId?: string) => {
      if (!dataSource) return;
      setIsSearching(true);
      try {
        const embedding = await dataSource.getImageEmbedding(
          imageId,
          flickrPhotoId,
        );
        if (!embedding) {
          setMessage("Embedding not found for this image.");
          return;
        }
        const events = eventNames ?? selectedEvents;
        const hits = await dataSource.searchByEmbedding(embedding, {
          limit: PAGE_SIZE,
          offset: 0,
          eventNames: events.length > 0 ? events : undefined,
          tagNames: selectedTags.length > 0 ? selectedTags : undefined,
        });
        setResults(hits);
        setCurrentEmbedding(embedding);
        setCurrentFaceEmbeddings(null);
        setOffset(PAGE_SIZE);
        setHasMore(hits.length === PAGE_SIZE);
        setMessage(`Found ${hits.length} similar images.`);
        if (eventNames) setSelectedEvents(eventNames);
      } finally {
        setIsSearching(false);
      }
    },
    [dataSource, selectedEvents, selectedTags],
  );

  const searchByFace = useCallback(
    async (faceEmbedding: number[], eventNames?: string[]) => {
      if (!dataSource) return;
      setIsSearching(true);
      try {
        const events = eventNames ?? selectedEvents;
        const hits = await dataSource.searchByFaceEmbedding(faceEmbedding, {
          limit: PAGE_SIZE,
          offset: 0,
          eventNames: events.length > 0 ? events : undefined,
          tagNames: selectedTags.length > 0 ? selectedTags : undefined,
          useVoronoi: !fullScan,
        });
        setResults(hits);
        setCurrentEmbedding(null);
        setCurrentFaceEmbeddings([faceEmbedding]);
        setOffset(PAGE_SIZE);
        setHasMore(hits.length === PAGE_SIZE);
        const mode = fullScan ? "(全件スキャン)" : "(Voronoi)";
        setMessage(`Found ${hits.length} images with similar faces. ${mode}`);
        if (eventNames) setSelectedEvents(eventNames);
      } finally {
        setIsSearching(false);
      }
    },
    [dataSource, selectedEvents, selectedTags, fullScan],
  );

  const searchByFaces = useCallback(
    async (
      faceEmbeddings: number[][],
      eventNames?: string[],
      overrideFullScan?: boolean,
    ) => {
      if (!dataSource || faceEmbeddings.length === 0) return;
      const isFullScan = overrideFullScan ?? fullScan;
      setIsSearching(true);
      try {
        const events = eventNames ?? selectedEvents;
        const evNames = events.length > 0 ? events : undefined;
        const tagNames = selectedTags.length > 0 ? selectedTags : undefined;
        const hits = await dataSource.searchByMultipleFaceEmbeddings(
          faceEmbeddings,
          {
            limit: PAGE_SIZE,
            offset: 0,
            eventNames: evNames,
            tagNames,
            useVoronoi: !isFullScan,
          },
        );
        setResults(hits);
        setCurrentEmbedding(null);
        setCurrentFaceEmbeddings(faceEmbeddings);
        setOffset(PAGE_SIZE);
        setHasMore(hits.length === PAGE_SIZE);
        const mode = isFullScan ? "(全件スキャン)" : "(Voronoi)";
        const msg =
          faceEmbeddings.length === 1
            ? `Found ${hits.length} images with similar faces. ${mode}`
            : `Found ${hits.length} images with all ${faceEmbeddings.length} faces. ${mode}`;
        setMessage(msg);
        if (eventNames) setSelectedEvents(eventNames);
      } finally {
        setIsSearching(false);
      }
    },
    [dataSource, selectedEvents, selectedTags, fullScan],
  );

  const loadMore = useCallback(async () => {
    if (!dataSource) return;
    const evNames = selectedEvents.length > 0 ? selectedEvents : undefined;
    const tagNames = selectedTags.length > 0 ? selectedTags : undefined;

    if (currentFaceEmbeddings) {
      // Face search load more
      setIsSearching(true);
      try {
        if (currentFaceEmbeddings.length === 1 && currentFaceEmbeddings[0]) {
          // Single face: offset-based pagination
          const hits = await dataSource.searchByFaceEmbedding(
            currentFaceEmbeddings[0],
            {
              limit: PAGE_SIZE,
              offset,
              eventNames: evNames,
              tagNames,
              useVoronoi: !fullScan,
            },
          );
          setResults((prev) => [...prev, ...hits]);
          setOffset((prev) => prev + hits.length);
          setHasMore(hits.length === PAGE_SIZE);
          setMessage(`Showing ${results.length + hits.length} images.`);
        } else {
          // Multi face: re-fetch with larger limit
          const newLimit = offset + PAGE_SIZE;
          const hits = await dataSource.searchByMultipleFaceEmbeddings(
            currentFaceEmbeddings,
            {
              limit: newLimit,
              offset: 0,
              eventNames: evNames,
              tagNames,
              useVoronoi: !fullScan,
            },
          );
          const hasNew = hits.length > results.length;
          setResults(hits);
          setOffset(newLimit);
          setHasMore(hasNew && hits.length === newLimit);
          setMessage(`Showing ${hits.length} images.`);
        }
      } finally {
        setIsSearching(false);
      }
      return;
    }

    if (!currentEmbedding) return;
    setIsSearching(true);
    try {
      const hits = await dataSource.searchByEmbedding(currentEmbedding, {
        limit: PAGE_SIZE,
        offset,
        eventNames: evNames,
        tagNames,
      });
      setResults((prev) => [...prev, ...hits]);
      setOffset((prev) => prev + hits.length);
      setHasMore(hits.length === PAGE_SIZE);
      setMessage(`Showing ${results.length + hits.length} images.`);
    } finally {
      setIsSearching(false);
    }
  }, [
    dataSource,
    currentEmbedding,
    currentFaceEmbeddings,
    offset,
    selectedEvents,
    selectedTags,
    results.length,
    fullScan,
  ]);

  return {
    results,
    hasMore,
    isSearching,
    message,
    selectedEvents,
    setSelectedEvents,
    selectedTags,
    setSelectedTags,
    fullScan,
    setFullScan,
    searchByText,
    searchByImage,
    searchByStoredEmbedding,
    searchByFace,
    searchByFaces,
    loadMore,
  };
}
