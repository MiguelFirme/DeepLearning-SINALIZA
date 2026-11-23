import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

// Executa o mesmo módulo usado pelo cliente, sem dependências de teste extras.
const source = readFileSync(new URL("../src/services/urls.ts", import.meta.url), "utf8");
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } }).outputText;
const { createApiUrls } = await import(`data:text/javascript;base64,${Buffer.from(js).toString("base64")}`);
const page = "https://sinaliza.vercel.app/dictionary";
const origin = "https://deeplearning-sinaliza.onrender.com";

for (const base of [origin, `${origin}/`, `${origin}/api`, `${origin}/api/`, ` ${origin}/api/ `]) {
  test(`HTTP, mídia e WebSocket com base ${base}`, () => {
    const urls = createApiUrls(base, page);
    assert.equal(urls.http("/health"), `${origin}/api/health`);
    assert.equal(urls.http("/api/dictionary?limit=12"), `${origin}/api/dictionary?limit=12`);
    assert.equal(urls.websocket(), "wss://deeplearning-sinaliza.onrender.com/api/ws/predict");
    const path = "/api/media/dictionary/Educa%C3%A7%C3%A3o/Escola.mp4";
    assert.equal(urls.media(path), `${origin}${path}`);
    assert.equal(urls.media("media/dictionary/Escola.mp4"), `${origin}/api/media/dictionary/Escola.mp4`);
    assert.equal(urls.media("api/media/dictionary/Escola.mp4"), `${origin}/api/media/dictionary/Escola.mp4`);
    assert.equal(urls.media("/api/media/capa.jpg"), `${origin}/api/media/capa.jpg`);
    assert.equal(urls.media("/api/media/face.png"), `${origin}/api/media/face.png`);
  });
}

test("sem variável usa o mesmo domínio e proxy Vite, sem porta inventada", () => {
  for (const base of [undefined, "", " "]) {
    const urls = createApiUrls(base, "http://localhost:5173/dictionary");
    assert.equal(urls.http("/health"), "http://localhost:5173/api/health");
    assert.equal(urls.websocket(), "ws://localhost:5173/api/ws/predict");
  }
  assert.equal(createApiUrls("/api/", page).websocket(), "wss://sinaliza.vercel.app/api/ws/predict");
});

test("mídia externa mantém a CDN, ausência não cria URL inválida", () => {
  const urls = createApiUrls(`${origin}/api`, page);
  assert.equal(urls.media("https://cdn.example.org/sinal.mp4"), "https://cdn.example.org/sinal.mp4");
  assert.equal(urls.media("//cdn.example.org/capa.jpg"), "https://cdn.example.org/capa.jpg");
  assert.equal(urls.media(null), null);
  assert.equal(urls.media(""), null);
});

test("prefixo de proxy e protocolo HTTP são preservados", () => {
  const urls = createApiUrls("http://localhost:8000/proxy/api/", page);
  assert.equal(urls.http("/health"), "http://localhost:8000/proxy/api/health");
  assert.equal(urls.websocket(), "ws://localhost:8000/proxy/api/ws/predict");
  assert.equal(urls.media("/api/media/test.mp4"), "http://localhost:8000/proxy/api/media/test.mp4");
});

test("recusa protocolo que não corresponde a uma API HTTP", () => {
  assert.throws(() => createApiUrls("ftp://example.org", page), /http ou https/);
});
