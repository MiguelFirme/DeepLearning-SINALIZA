/** Verifica o build real no Edge headless, com catálogo local e rolagem. Node 22+. */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, existsSync } from "node:fs";
import { resolve, join, sep, extname } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const output = resolve(root, "logs/sinalario_responsivo");
mkdirSync(output, { recursive: true });
const catalog = JSON.parse(readFileSync(join(output, "catalogo.json"), "utf8"));
const categories = [...new Set(catalog.map((entry) => entry.category))];
const browserPath = process.env.BROWSER_PATH || "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
assert.ok(existsSync(browserPath), "Informe BROWSER_PATH com o caminho completo do navegador Chromium");
const types = { ".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg", ".png": "image/png", ".mp4": "video/mp4" };
const server = createServer((req, res) => {
  try {
    const path = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    const media = path.startsWith("/api/media/dictionary/");
    const base = resolve(root, media ? "backend/static/dictionary" : "frontend/dist");
    const relative = media ? path.slice("/api/media/dictionary/".length) : path.slice(1);
    let file = resolve(base, relative || "index.html");
    if (!file.startsWith(base + sep)) throw new Error("Caminho fora da pasta servida");
    if (!existsSync(file) && !media) file = join(base, "index.html");
    res.setHeader("Content-Type", types[extname(file)] || "application/octet-stream");
    res.end(readFileSync(file));
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const origin = `http://127.0.0.1:${server.address().port}`;
const profile = mkdtempSync(join(tmpdir(), "sinaliza-layout-"));
const browser = spawn(browserPath, ["--headless=new", "--disable-gpu", "--no-first-run",
  "--no-default-browser-check", "--remote-debugging-port=0", `--user-data-dir=${profile}`, "about:blank"],
  { windowsHide: true, stdio: "ignore" });
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));
let ws;
try {
  const activePort = join(profile, "DevToolsActivePort");
  for (let i = 0; i < 100 && !existsSync(activePort); i++) await sleep(100);
  const port = readFileSync(activePort, "utf8").split(/\r?\n/)[0];
  const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  ws = new WebSocket(pages.find((page) => page.type === "page").webSocketDebuggerUrl);
  await new Promise((done, reject) => { ws.onopen = done; ws.onerror = reject; });
  let nextId = 0;
  const pending = new Map();
  function call(method, params = {}) {
    const id = ++nextId;
    return new Promise((done, reject) => {
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timeout: ${method}`)); }, 15000);
      pending.set(id, { done: (value) => { clearTimeout(timer); done(value); },
        reject: (error) => { clearTimeout(timer); reject(error); } });
      ws.send(JSON.stringify({ id, method, params }));
    });
  }
  const errors = [];
  ws.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.id) {
      const item = pending.get(message.id);
      if (item) { pending.delete(message.id); message.error ? item.reject(message.error) : item.done(message.result); }
    } else if (message.method === "Fetch.requestPaused") {
      const { request, requestId } = message.params;
      const url = new URL(request.url);
      let entries = catalog.filter((entry) => !url.searchParams.get("category") || entry.category === url.searchParams.get("category"));
      const total = entries.length;
      const offset = Number(url.searchParams.get("offset") || 0);
      entries = entries.slice(offset, offset + Number(url.searchParams.get("limit") || 50)).map((entry) => {
        const result = { ...entry };
        for (const field of ["image_url", "poster_url", "example_video_url"]) {
          if (result[field]) result[field] = origin + result[field];
        }
        return result;
      });
      call("Fetch.fulfillRequest", { requestId, responseCode: request.method === "OPTIONS" ? 204 : 200,
        responseHeaders: [{ name: "Content-Type", value: "application/json" },
          { name: "Access-Control-Allow-Origin", value: origin },
          { name: "Access-Control-Allow-Methods", value: "GET, OPTIONS" },
          { name: "Access-Control-Allow-Headers", value: "content-type" }],
        body: request.method === "OPTIONS" ? "" : Buffer.from(JSON.stringify({ entries, total, categories })).toString("base64"),
      }).catch((error) => errors.push(String(error)));
    } else if (message.method === "Runtime.exceptionThrown") {
      errors.push(message.params.exceptionDetails.text);
    }
  };
  const evaluate = async (expression) => {
    const result = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  await call("Runtime.enable");
  await call("Page.enable");
  await call("Fetch.enable", { patterns: [{ urlPattern: "*onrender.com/api/dictionary*" }] });
  const results = [];
  for (const width of [320, 390, 768, 1024, 1440]) {
    const mobile = width <= 768;
    await call("Emulation.setDeviceMetricsOverride", { width, height: 850, deviceScaleFactor: 1, mobile });
    await call("Emulation.setTouchEmulationEnabled", { enabled: mobile });
    await call("Page.navigate", { url: `${origin}/dictionary` });
    for (let i = 0; i < 80; i++) {
      if (await evaluate("Boolean(document.querySelector('aside button'))")) break;
      await sleep(100);
    }
    await evaluate("[...document.querySelectorAll('aside button')].find(b => b.textContent.includes('Educação')).click()");
    for (let i = 0; i < 80; i++) {
      if (await evaluate("Boolean(document.querySelector('section video'))")) break;
      await sleep(100);
    }
    await sleep(200);
    const metrics = () => evaluate(`(() => {
      const box = s => { const r = document.querySelector(s).getBoundingClientRect(); return {top:r.top,bottom:r.bottom,left:r.left,right:r.right}; };
      return {width:innerWidth,scrollWidth:document.documentElement.scrollWidth,
        sidebar:box('aside'),main:box('section'),header:box('header'),title:box('h1'),
        position:getComputedStyle(document.querySelector('aside')).position,
        links:[...document.querySelectorAll('header nav a')].map(a=>{const r=a.getBoundingClientRect(); return {left:r.left,right:r.right,bottom:r.bottom};}),
        videos:document.querySelectorAll('section video').length};
    })()`);
    const before = await metrics();
    assert.ok(before.videos > 0, "O catálogo de vídeos não carregou");
    assert.ok(before.scrollWidth <= width, `Overflow horizontal em ${width}`);
    assert.ok(before.title.top >= before.header.bottom, `Título sob o cabeçalho em ${width}`);
    assert.ok(before.links.every((link) => link.left >= 0 && link.right <= width && link.bottom <= before.header.bottom), `Menu cortado em ${width}`);
    if (mobile) {
      assert.equal(before.position, "static");
      assert.ok(before.sidebar.bottom <= before.main.top, `Categorias sobrepostas antes de rolar em ${width}`);
    } else {
      assert.equal(before.position, "sticky");
      assert.ok(before.sidebar.right <= before.main.left, `Colunas sobrepostas em ${width}`);
    }
    const shot = async (name) => {
      const { data } = await call("Page.captureScreenshot", { format: "png" });
      writeFileSync(join(output, name), Buffer.from(data, "base64"));
    };
    await shot(`${width}-inicio.png`);
    await evaluate("scrollTo(0,600)");
    await sleep(150);
    const after = await metrics();
    assert.ok(mobile ? after.sidebar.bottom <= after.main.top : after.sidebar.right <= after.main.left, `Sobreposição ao rolar em ${width}`);
    await shot(`${width}-rolagem.png`);
    results.push({ width, before, after });
    console.log(`${width}px: sem overflow, menu visível, categorias separadas antes e depois da rolagem`);
  }
  assert.deepEqual(errors, []);
  writeFileSync(join(output, "resultado.json"), JSON.stringify(results, null, 2));
} finally {
  ws?.close();
  browser.kill();
  server.closeAllConnections();
  server.close();
}
