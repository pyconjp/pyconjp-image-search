import * as duckdb from "@duckdb/duckdb-wasm";
import duckdb_wasm from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import duckdb_wasm_eh from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import duckdb_worker from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import duckdb_worker_eh from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url";

let dbInstance: duckdb.AsyncDuckDB | null = null;
let connInstance: duckdb.AsyncDuckDBConnection | null = null;

const MANUAL_BUNDLES: duckdb.DuckDBBundles = {
  mvp: {
    mainModule: duckdb_wasm,
    mainWorker: duckdb_worker,
  },
  eh: {
    mainModule: duckdb_wasm_eh,
    mainWorker: duckdb_worker_eh,
  },
};

export async function initDuckDB(): Promise<duckdb.AsyncDuckDBConnection> {
  if (connInstance) return connInstance;

  const bundle = await duckdb.selectBundle(MANUAL_BUNDLES);

  const worker = new Worker(bundle.mainWorker!);
  const logger = new duckdb.ConsoleLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);

  // Fetch the pre-built CLIP DuckDB file
  const response = await fetch("/pyconjp_image_search_clip.duckdb");
  const buffer = await response.arrayBuffer();
  await db.registerFileBuffer(
    "pyconjp_image_search_clip.duckdb",
    new Uint8Array(buffer),
  );

  const conn = await db.connect();
  await conn.query(
    "ATTACH 'pyconjp_image_search_clip.duckdb' AS data (READ_ONLY)",
  );

  dbInstance = db;
  connInstance = conn;
  return conn;
}

export function getConnection(): duckdb.AsyncDuckDBConnection | null {
  return connInstance;
}

export { dbInstance };
