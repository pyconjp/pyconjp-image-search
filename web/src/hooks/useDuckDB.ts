import type { AsyncDuckDBConnection } from "@duckdb/duckdb-wasm";
import { useEffect, useRef, useState } from "react";
import { closeDuckDB, type DuckDBInstance, initDuckDB } from "../lib/duckdb";

export function useDuckDB(dbFileName: string | null) {
  const [conn, setConn] = useState<AsyncDuckDBConnection | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const instanceRef = useRef<DuckDBInstance | null>(null);

  useEffect(() => {
    if (!dbFileName) {
      setConn(null);
      setIsLoading(false);
      return;
    }

    let cancelled = false;
    setIsLoading(true);
    setError(null);
    setConn(null);

    // Close previous instance
    const prevInstance = instanceRef.current;
    if (prevInstance) {
      instanceRef.current = null;
      closeDuckDB(prevInstance);
    }

    initDuckDB(dbFileName)
      .then((instance) => {
        if (!cancelled) {
          instanceRef.current = instance;
          setConn(instance.conn);
          setIsLoading(false);
        } else {
          closeDuckDB(instance);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [dbFileName]);

  return { conn, isLoading, error };
}
