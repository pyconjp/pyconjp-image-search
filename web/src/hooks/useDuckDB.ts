import { useEffect, useState } from "react";
import type { AsyncDuckDBConnection } from "@duckdb/duckdb-wasm";
import { initDuckDB } from "../lib/duckdb";

export function useDuckDB() {
  const [conn, setConn] = useState<AsyncDuckDBConnection | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    initDuckDB()
      .then((c) => {
        if (!cancelled) {
          setConn(c);
          setIsLoading(false);
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
  }, []);

  return { conn, isLoading, error };
}
