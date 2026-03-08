import type { AsyncDuckDBConnection } from "@duckdb/duckdb-wasm";
import { useMemo } from "react";
import type { DataSource } from "../lib/datasource";
import { DuckDBDataSource } from "../lib/duckdb-datasource";
import { FirestoreDataSourceExtended } from "../lib/firestore-datasource";
import type { ModelConfig } from "../lib/models";

const DATASOURCE_TYPE = import.meta.env.VITE_DATASOURCE ?? "duckdb";

export function useDataSource(
  conn: AsyncDuckDBConnection | null,
  config: ModelConfig,
): DataSource | null {
  return useMemo(() => {
    if (DATASOURCE_TYPE === "firestore") {
      return new FirestoreDataSourceExtended();
    }
    // DuckDB mode: need connection
    if (!conn) return null;
    return new DuckDBDataSource(conn, config);
  }, [conn, config]);
}

export { DATASOURCE_TYPE };
