import { useState, useCallback } from "react";
import type { AsyncDuckDBConnection } from "@duckdb/duckdb-wasm";
import type { SearchResult } from "../types";
import { searchByEmbedding, getImageEmbedding } from "../lib/search";
import type { CLIPEncoder } from "../lib/clip";

const PAGE_SIZE = 20;

export function useImageSearch(
  conn: AsyncDuckDBConnection | null,
  encoder: CLIPEncoder | null,
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

  const searchByText = useCallback(
    async (query: string, eventNames?: string[]) => {
      if (!conn || !encoder) return;
      if (!query.trim()) {
        setMessage("Please enter a search query.");
        return;
      }
      setIsSearching(true);
      try {
        const embedding = await encoder.encodeText(query);
        const events = eventNames ?? selectedEvents;
        const hits = await searchByEmbedding(conn, embedding, {
          limit: PAGE_SIZE,
          offset: 0,
          eventNames: events.length > 0 ? events : undefined,
        });
        setResults(hits);
        setCurrentEmbedding(embedding);
        setOffset(PAGE_SIZE);
        setHasMore(hits.length === PAGE_SIZE);
        setMessage(`Found ${hits.length} images for "${query}".`);
        if (eventNames) setSelectedEvents(eventNames);
      } finally {
        setIsSearching(false);
      }
    },
    [conn, encoder, selectedEvents],
  );

  const searchByImage = useCallback(
    async (imageBlob: Blob, eventNames?: string[]) => {
      if (!conn || !encoder) return;
      setIsSearching(true);
      try {
        const embedding = await encoder.encodeImage(imageBlob);
        const events = eventNames ?? selectedEvents;
        const hits = await searchByEmbedding(conn, embedding, {
          limit: PAGE_SIZE,
          offset: 0,
          eventNames: events.length > 0 ? events : undefined,
        });
        setResults(hits);
        setCurrentEmbedding(embedding);
        setOffset(PAGE_SIZE);
        setHasMore(hits.length === PAGE_SIZE);
        setMessage(`Found ${hits.length} similar images.`);
        if (eventNames) setSelectedEvents(eventNames);
      } finally {
        setIsSearching(false);
      }
    },
    [conn, encoder, selectedEvents],
  );

  const searchByStoredEmbedding = useCallback(
    async (imageId: number, eventNames?: string[]) => {
      if (!conn) return;
      setIsSearching(true);
      try {
        const embedding = await getImageEmbedding(conn, imageId);
        if (!embedding) {
          setMessage("Embedding not found for this image.");
          return;
        }
        const events = eventNames ?? selectedEvents;
        const hits = await searchByEmbedding(conn, embedding, {
          limit: PAGE_SIZE,
          offset: 0,
          eventNames: events.length > 0 ? events : undefined,
        });
        setResults(hits);
        setCurrentEmbedding(embedding);
        setOffset(PAGE_SIZE);
        setHasMore(hits.length === PAGE_SIZE);
        setMessage(`Found ${hits.length} similar images.`);
        if (eventNames) setSelectedEvents(eventNames);
      } finally {
        setIsSearching(false);
      }
    },
    [conn, selectedEvents],
  );

  const loadMore = useCallback(async () => {
    if (!conn || !currentEmbedding) return;
    setIsSearching(true);
    try {
      const hits = await searchByEmbedding(conn, currentEmbedding, {
        limit: PAGE_SIZE,
        offset,
        eventNames: selectedEvents.length > 0 ? selectedEvents : undefined,
      });
      setResults((prev) => [...prev, ...hits]);
      setOffset((prev) => prev + hits.length);
      setHasMore(hits.length === PAGE_SIZE);
      setMessage(`Showing ${results.length + hits.length} images.`);
    } finally {
      setIsSearching(false);
    }
  }, [conn, currentEmbedding, offset, selectedEvents, results.length]);

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
    loadMore,
  };
}
