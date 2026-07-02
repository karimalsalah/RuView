/**
 * Streamable HTTP transport for @ruvnet/rvagent (ADR-124 §3, hardened per
 * ADR-264 F7/O3).
 *
 * Binds to 127.0.0.1 by default and mounts an /mcp endpoint backed by
 * StreamableHTTPServerTransport from @modelcontextprotocol/sdk.
 *
 * Session model (ADR-264 F7): the SDK's stateful mode requires ONE transport
 * (and one MCP Server) per session. An `initialize` POST creates a fresh
 * transport + server pair via the caller-supplied factory; follow-up
 * POST/GET/DELETE requests are routed to their session by the
 * `mcp-session-id` header. Transports are dropped when their session closes.
 *
 * Security model (ADR-124 §6 + ADR-264 F7):
 *   - Origin validation: browser-style requests whose Origin is not local
 *     are rejected with 403 before reaching the MCP layer. Localhost origins
 *     match on hostname, ANY port (http://localhost:5173 is local).
 *   - Bearer token: when RVAGENT_HTTP_TOKEN is set, requests must carry
 *     Authorization: Bearer <token>; missing/wrong tokens → 401.
 *   - Body cap: request bodies over 1 MiB are rejected with 413 (the
 *     unbounded-buffering DoS from the pre-ADR-264 scaffold).
 *   - Bind address: defaults to 127.0.0.1 per MCP spec security requirement.
 *     Set RVAGENT_HTTP_HOST=0.0.0.0 only for intentional fleet deployment.
 *
 * Usage:
 *   import { createHttpTransport } from './http-transport.js';
 *   const { httpServer } = await createHttpTransport(() => buildServer(config));
 *   // httpServer is a node:http.Server — call httpServer.close() to shut down.
 */

import { createServer, type Server as HttpServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import type { Server as McpServer } from "@modelcontextprotocol/sdk/server/index.js";

export type McpServerFactory = () => McpServer;

export interface HttpTransportOptions {
  /** TCP host to bind (default: 127.0.0.1). */
  host?: string;
  /** TCP port to listen on (default: 3001). */
  port?: number;
  /**
   * Allowed Origin header values. Requests with an Origin not in this list
   * (and not a localhost origin) are rejected with 403. Use '*' to disable
   * Origin validation entirely (not recommended outside of local-dev flags).
   */
  allowedOrigins?: string[];
  /**
   * Bearer token for HTTP transport. When set, every request must supply
   * Authorization: Bearer <token>; omitted or wrong token → 401.
   * Defaults to process.env.RVAGENT_HTTP_TOKEN (undefined = auth disabled).
   */
  bearerToken?: string;
  /** Maximum accepted request body size in bytes (default: 1 MiB). */
  maxBodyBytes?: number;
}

export interface HttpTransportResult {
  /** The raw Node.js HTTP server — call .close() to shut down. */
  httpServer: HttpServer;
  /** Live sessions keyed by session id (exposed for tests/observability). */
  sessions: Map<string, StreamableHTTPServerTransport>;
  /** The bound address string (e.g. "http://127.0.0.1:3001"). */
  boundAddress: string;
}

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 3001;
const DEFAULT_MAX_BODY_BYTES = 1024 * 1024;
const LOCAL_HOSTNAMES = new Set(["localhost", "127.0.0.1", "[::1]"]);

/**
 * Validate Origin header against the allowlist.
 * Returns true if the request should be allowed, false if it should be rejected.
 *
 * An absent Origin header is allowed (same-origin non-browser requests, curl,
 * etc.). A localhost origin is allowed on any port (real browser origins carry
 * ports — ADR-264 F7). Anything else must match the allowlist exactly.
 */
export function isOriginAllowed(
  origin: string | undefined,
  allowedOrigins: string[]
): boolean {
  if (origin === undefined) return true; // no Origin = not a cross-origin browser request
  if (allowedOrigins.includes("*")) return true;
  if (allowedOrigins.includes(origin)) return true;
  try {
    const u = new URL(origin);
    return (
      (u.protocol === "http:" || u.protocol === "https:") &&
      LOCAL_HOSTNAMES.has(u.hostname === "::1" ? "[::1]" : u.hostname)
    );
  } catch {
    return false;
  }
}

/** Read a request body with a hard size cap; null = payload too large. */
function readBody(
  req: IncomingMessage,
  maxBytes: number
): Promise<string | null> {
  return new Promise((resolve, reject) => {
    let size = 0;
    let tooLarge = false;
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => {
      if (tooLarge) return; // keep draining so the 413 response can flush
      size += chunk.length;
      if (size > maxBytes) {
        tooLarge = true;
        chunks.length = 0;
        resolve(null);
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (!tooLarge) resolve(Buffer.concat(chunks).toString("utf8"));
    });
    req.on("error", reject);
  });
}

function json(res: ServerResponse, status: number, body: object): void {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

/**
 * Build the HTTP server around a per-session MCP transport map.
 * Returns the Node.js HTTP server (not yet listening) plus the session map.
 * Call httpServer.listen(port, host) or rely on createHttpTransport which
 * does that for you.
 */
export function buildHttpApp(
  serverFactory: McpServerFactory,
  opts: HttpTransportOptions = {}
): { httpServer: HttpServer; sessions: Map<string, StreamableHTTPServerTransport> } {
  const allowedOrigins: string[] = opts.allowedOrigins ?? [];
  const bearerToken = opts.bearerToken ?? process.env["RVAGENT_HTTP_TOKEN"];
  const maxBodyBytes = opts.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
  const sessions = new Map<string, StreamableHTTPServerTransport>();

  const httpServer = createServer((req: IncomingMessage, res: ServerResponse) => {
    void (async () => {
      // ── Origin validation ──────────────────────────────────────────────
      const origin = req.headers["origin"] as string | undefined;
      if (!isOriginAllowed(origin, allowedOrigins)) {
        json(res, 403, { error: "Forbidden: cross-origin request rejected" });
        return;
      }

      // ── Bearer token auth ──────────────────────────────────────────────
      if (bearerToken !== undefined && bearerToken !== "") {
        const authHeader = req.headers["authorization"] as string | undefined;
        const supplied = authHeader?.startsWith("Bearer ")
          ? authHeader.slice("Bearer ".length)
          : undefined;
        if (supplied !== bearerToken) {
          json(res, 401, { error: "Unauthorized: missing or invalid bearer token" });
          return;
        }
      }

      // ── Route: /mcp ────────────────────────────────────────────────────
      if (req.url !== "/mcp") {
        json(res, 404, { error: "Not found. MCP endpoint: /mcp" });
        return;
      }

      const sessionId = req.headers["mcp-session-id"] as string | undefined;

      if (req.method === "POST") {
        const body = await readBody(req, maxBodyBytes);
        if (body === null) {
          json(res, 413, { error: `Payload too large (max ${maxBodyBytes} bytes)` });
          return;
        }
        let parsed: unknown;
        try {
          parsed = JSON.parse(body);
        } catch {
          json(res, 400, { error: "Bad Request: invalid JSON body" });
          return;
        }

        // Existing session → route to its transport.
        if (sessionId !== undefined) {
          const transport = sessions.get(sessionId);
          if (!transport) {
            json(res, 404, { error: `Unknown session "${sessionId}"` });
            return;
          }
          await transport.handleRequest(req, res, parsed);
          return;
        }

        // New session: must be an initialize request (ADR-264 F7 — one
        // transport + one MCP Server per session).
        if (!isInitializeRequest(parsed)) {
          json(res, 400, {
            error: "Bad Request: no mcp-session-id and not an initialize request",
          });
          return;
        }
        const transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: () => randomUUID(),
          onsessioninitialized: (id: string) => {
            sessions.set(id, transport);
          },
        });
        transport.onclose = () => {
          if (transport.sessionId !== undefined) sessions.delete(transport.sessionId);
        };
        const mcpServer = serverFactory();
        await mcpServer.connect(transport as Parameters<typeof mcpServer.connect>[0]);
        await transport.handleRequest(req, res, parsed);
        return;
      }

      // GET (SSE stream) / DELETE (session termination) — session-scoped.
      if (req.method === "GET" || req.method === "DELETE") {
        const transport = sessionId !== undefined ? sessions.get(sessionId) : undefined;
        if (!transport) {
          json(res, 400, { error: "Bad Request: missing or unknown mcp-session-id" });
          return;
        }
        await transport.handleRequest(req, res);
        return;
      }

      json(res, 405, { error: "Method not allowed. Use POST/GET/DELETE on /mcp" });
    })().catch(() => {
      if (!res.headersSent) json(res, 500, { error: "Internal server error" });
      else res.end();
    });
  });

  return { httpServer, sessions };
}

/**
 * Create and start the Streamable HTTP transport, resolving once the server
 * is bound and listening.
 */
export async function createHttpTransport(
  serverFactory: McpServerFactory,
  opts: HttpTransportOptions = {}
): Promise<HttpTransportResult> {
  const host = opts.host ?? process.env["RVAGENT_HTTP_HOST"] ?? DEFAULT_HOST;
  const port = opts.port ?? Number(process.env["RVAGENT_HTTP_PORT"] ?? DEFAULT_PORT);

  const { httpServer, sessions } = buildHttpApp(serverFactory, opts);

  await new Promise<void>((resolve, reject) => {
    httpServer.once("error", reject);
    httpServer.listen(port, host, () => resolve());
  });

  return {
    httpServer,
    sessions,
    boundAddress: `http://${host}:${port}`,
  };
}
