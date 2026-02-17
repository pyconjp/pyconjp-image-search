import * as duckdb from "@duckdb/duckdb-wasm";
import duckdb_worker_eh from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url";
import duckdb_worker from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import duckdb_wasm_eh from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import duckdb_wasm from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";

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

export interface DuckDBInstance {
  db: duckdb.AsyncDuckDB;
  conn: duckdb.AsyncDuckDBConnection;
}

export async function initDuckDB(dbFileName: string): Promise<DuckDBInstance> {
  const bundle = await duckdb.selectBundle(MANUAL_BUNDLES);

  const worker = new Worker(bundle.mainWorker!);
  const logger = new duckdb.VoidLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);

  const response = await fetch(`/${dbFileName}`);
  const buffer = await response.arrayBuffer();
  await db.registerFileBuffer(dbFileName, new Uint8Array(buffer));

  const conn = await db.connect();
  await conn.query(`ATTACH '${dbFileName}' AS data (READ_ONLY)`);

  return { db, conn };
}

export async function closeDuckDB(instance: DuckDBInstance): Promise<void> {
  try {
    await instance.conn.close();
  } catch {
    // ignore close errors
  }
  try {
    await instance.db.terminate();
  } catch {
    // ignore terminate errors
  }
}
