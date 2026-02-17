import type { AsyncDuckDBConnection } from "@duckdb/duckdb-wasm";
import { useCallback, useState } from "react";
import type { VisionLanguageEncoder } from "../lib/encoder";
import type { ModelConfig } from "../lib/models";
import {
  getImageEmbedding,
  type SearchConfig,
  searchByEmbedding,
  searchByFaceEmbedding,
  searchByMultipleFaceEmbeddings,
} from "../lib/search";
import type { SearchResult } from "../types";

const PAGE_SIZE = 20;

export function useImageSearch(
  conn: AsyncDuckDBConnection | null,
  encoder: VisionLanguageEncoder | null,
  modelConfig: ModelConfig | null,
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
  const [currentFaceEmbeddings, setCurrentFaceEmbeddings] = useState<
    number[][] | null
  >(null);

  const getSearchConfig = useCallback((): SearchConfig | null => {
    if (!modelConfig) return null;
    return {
      modelName: modelConfig.modelName,
      embeddingDim: modelConfig.embeddingDim,
    };
  }, [modelConfig]);

  const searchByText = useCallback(
    async (query: string, eventNames?: string[]) => {
      if (!conn || !encoder) return;
      const config = getSearchConfig();
      if (!config) return;
      if (!query.trim()) {
        setMessage("Please enter a search query.");
        return;
      }
      setIsSearching(true);
      try {
        const embedding = await encoder.encodeText(query);
        const events = eventNames ?? selectedEvents;
        const hits = await searchByEmbedding(
          conn,
          embedding,
          {
            limit: PAGE_SIZE,
            offset: 0,
            eventNames: events.length > 0 ? events : undefined,
          },
          config,
        );
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
    [conn, encoder, selectedEvents, getSearchConfig],
  );

  const searchByImage = useCallback(
    async (imageBlob: Blob, eventNames?: string[]) => {
      if (!conn || !encoder) return;
      const config = getSearchConfig();
      if (!config) return;
      setIsSearching(true);
      try {
        const embedding = await encoder.encodeImage(imageBlob);
        const events = eventNames ?? selectedEvents;
        const hits = await searchByEmbedding(
          conn,
          embedding,
          {
            limit: PAGE_SIZE,
            offset: 0,
            eventNames: events.length > 0 ? events : undefined,
          },
          config,
        );
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
    [conn, encoder, selectedEvents, getSearchConfig],
  );

  const searchByStoredEmbedding = useCallback(
    async (imageId: number, eventNames?: string[]) => {
      if (!conn || !modelConfig) return;
      const config = getSearchConfig();
      if (!config) return;
      setIsSearching(true);
      try {
        const embedding = await getImageEmbedding(
          conn,
          imageId,
          modelConfig.modelName,
        );
        if (!embedding) {
          setMessage("Embedding not found for this image.");
          return;
        }
        const events = eventNames ?? selectedEvents;
        const hits = await searchByEmbedding(
          conn,
          embedding,
          {
            limit: PAGE_SIZE,
            offset: 0,
            eventNames: events.length > 0 ? events : undefined,
          },
          config,
        );
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
    [conn, modelConfig, selectedEvents, getSearchConfig],
  );

  const searchByFace = useCallback(
    async (faceEmbedding: number[], eventNames?: string[]) => {
      if (!conn) return;
      setIsSearching(true);
      try {
        const events = eventNames ?? selectedEvents;
        const hits = await searchByFaceEmbedding(conn, faceEmbedding, {
          limit: PAGE_SIZE,
          offset: 0,
          eventNames: events.length > 0 ? events : undefined,
        });
        setResults(hits);
        setCurrentEmbedding(null);
        setCurrentFaceEmbeddings([faceEmbedding]);
        setOffset(PAGE_SIZE);
        setHasMore(hits.length === PAGE_SIZE);
        setMessage(`Found ${hits.length} images with similar faces.`);
        if (eventNames) setSelectedEvents(eventNames);
      } finally {
        setIsSearching(false);
      }
    },
    [conn, selectedEvents],
  );

  const searchByFaces = useCallback(
    async (faceEmbeddings: number[][], eventNames?: string[]) => {
      if (!conn || faceEmbeddings.length === 0) return;
      setIsSearching(true);
      try {
        const events = eventNames ?? selectedEvents;
        const evNames = events.length > 0 ? events : undefined;
        const hits = await searchByMultipleFaceEmbeddings(
          conn,
          faceEmbeddings,
          {
            limit: PAGE_SIZE,
            offset: 0,
            eventNames: evNames,
          },
        );
        setResults(hits);
        setCurrentEmbedding(null);
        setCurrentFaceEmbeddings(faceEmbeddings);
        setOffset(PAGE_SIZE);
        setHasMore(hits.length === PAGE_SIZE);
        const msg =
          faceEmbeddings.length === 1
            ? `Found ${hits.length} images with similar faces.`
            : `Found ${hits.length} images with all ${faceEmbeddings.length} faces.`;
        setMessage(msg);
        if (eventNames) setSelectedEvents(eventNames);
      } finally {
        setIsSearching(false);
      }
    },
    [conn, selectedEvents],
  );

  const loadMore = useCallback(async () => {
    if (!conn) return;
    const config = getSearchConfig();
    const evNames = selectedEvents.length > 0 ? selectedEvents : undefined;

    if (currentFaceEmbeddings) {
      // Face search load more
      setIsSearching(true);
      try {
        if (currentFaceEmbeddings.length === 1 && currentFaceEmbeddings[0]) {
          // Single face: offset-based pagination
          const hits = await searchByFaceEmbedding(
            conn,
            currentFaceEmbeddings[0],
            { limit: PAGE_SIZE, offset, eventNames: evNames },
          );
          setResults((prev) => [...prev, ...hits]);
          setOffset((prev) => prev + hits.length);
          setHasMore(hits.length === PAGE_SIZE);
          setMessage(`Showing ${results.length + hits.length} images.`);
        } else {
          // Multi face: re-fetch with larger limit
          const newLimit = offset + PAGE_SIZE;
          const hits = await searchByMultipleFaceEmbeddings(
            conn,
            currentFaceEmbeddings,
            { limit: newLimit, offset: 0, eventNames: evNames },
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

    if (!currentEmbedding || !config) return;
    setIsSearching(true);
    try {
      const hits = await searchByEmbedding(
        conn,
        currentEmbedding,
        {
          limit: PAGE_SIZE,
          offset,
          eventNames: evNames,
        },
        config,
      );
      setResults((prev) => [...prev, ...hits]);
      setOffset((prev) => prev + hits.length);
      setHasMore(hits.length === PAGE_SIZE);
      setMessage(`Showing ${results.length + hits.length} images.`);
    } finally {
      setIsSearching(false);
    }
  }, [
    conn,
    currentEmbedding,
    currentFaceEmbeddings,
    offset,
    selectedEvents,
    results.length,
    getSearchConfig,
  ]);

  return {
    results,
    hasMore,
    isSearching,
    message,
    selectedEvents,
    setSelectedEvents,
    searchByText,
    searchByImage,
    searchByStoredEmbedding,
    searchByFace,
    searchByFaces,
    loadMore,
  };
}
