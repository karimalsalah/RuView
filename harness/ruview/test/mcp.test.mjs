// SPDX-License-Identifier: MIT
// MCP stdio server e2e — spawns `bin/cli.js mcp start` and speaks JSON-RPC.
// Pins ADR-263 O2 (ping answered while a long tools/call runs), O6 (version
// from package.json), and O8 (underscore names advertised, dotted accepted,
// resources/prompts stubs).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { which } from '../src/tools.js';

const PKG_ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const CLI = join(PKG_ROOT, 'bin', 'cli.js');

/** Start the MCP server; returns {send, next, close} where next(id) resolves the response with that id. */
function startServer() {
  const child = spawn(process.execPath, [CLI, 'mcp', 'start'], { stdio: ['pipe', 'pipe', 'pipe'] });
  const waiters = new Map();
  let buf = '';
  child.stdout.on('data', (d) => {
    buf += d;
    let nl;
    while ((nl = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const msg = JSON.parse(line);
      const w = waiters.get(msg.id);
      if (w) { waiters.delete(msg.id); w(msg); }
    }
  });
  return {
    send(msg) { child.stdin.write(JSON.stringify(msg) + '\n'); },
    next(id) { return new Promise((res) => waiters.set(id, res)); },
    close() { child.stdin.end(); child.kill(); },
  };
}

test('MCP handshake: initialize reports the package.json version; list endpoints respond', async () => {
  const pkg = JSON.parse(readFileSync(join(PKG_ROOT, 'package.json'), 'utf8'));
  const s = startServer();
  try {
    s.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: {} });
    const init = await s.next(1);
    assert.equal(init.result.serverInfo.version, pkg.version, 'ADR-263 O6: version must match package.json');

    s.send({ jsonrpc: '2.0', id: 2, method: 'tools/list' });
    const tools = (await s.next(2)).result.tools;
    assert.equal(tools.length, 6);
    for (const t of tools) assert.match(t.name, /^[a-zA-Z0-9_-]{1,64}$/, `advertised name not host-safe: ${t.name}`);

    s.send({ jsonrpc: '2.0', id: 3, method: 'resources/list' });
    assert.deepEqual((await s.next(3)).result, { resources: [] });
    s.send({ jsonrpc: '2.0', id: 4, method: 'prompts/list' });
    assert.deepEqual((await s.next(4)).result, { prompts: [] });

    // Dotted legacy name still callable (alias).
    s.send({ jsonrpc: '2.0', id: 5, method: 'tools/call', params: { name: 'ruview.onboard', arguments: {} } });
    const call = await s.next(5);
    assert.equal(call.result.isError, false);
  } finally {
    s.close();
  }
});

test('MCP server answers ping while a long tools/call is in flight (ADR-263 O2)', { skip: !which('python') && !which('python3') ? 'python not on PATH' : false }, async () => {
  // Fake RuView repo whose verify.py sleeps 3 s then passes.
  const repo = mkdtempSync(join(tmpdir(), 'ruview-mcp-e2e-'));
  const proofDir = join(repo, 'archive', 'v1', 'data', 'proof');
  mkdirSync(proofDir, { recursive: true });
  writeFileSync(join(proofDir, 'verify.py'), 'import time\ntime.sleep(3)\nprint("VERDICT: PASS")\n');

  const s = startServer();
  try {
    s.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: {} });
    await s.next(1);

    const verifyDone = s.next(10);
    s.send({ jsonrpc: '2.0', id: 10, method: 'tools/call', params: { name: 'ruview_verify', arguments: { repo } } });

    // Give the server a beat to start the child, then ping.
    await new Promise((r) => setTimeout(r, 300));
    const t0 = Date.now();
    const pinged = s.next(11);
    s.send({ jsonrpc: '2.0', id: 11, method: 'ping' });
    await pinged;
    const pingMs = Date.now() - t0;
    assert.ok(pingMs < 1000, `ping took ${pingMs} ms while verify was in flight — server is blocking`);

    const verify = await verifyDone;
    const payload = JSON.parse(verify.result.content[0].text);
    assert.equal(payload.verdict, 'PASS');
  } finally {
    s.close();
    rmSync(repo, { recursive: true, force: true });
  }
});
