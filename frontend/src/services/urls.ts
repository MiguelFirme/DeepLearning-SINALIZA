/** URLs compartilhadas por HTTP, WebSocket e mídia, inclusive em domínios separados. */
export function createApiUrls(configuredBase: string | undefined, pageUrl: string) {
  const base = new URL(configuredBase?.trim() || "/api", pageUrl);
  if (base.protocol !== "http:" && base.protocol !== "https:") {
    throw new Error("VITE_API_URL deve usar http ou https.");
  }
  base.search = "";
  base.hash = "";
  const pathname = base.pathname.replace(/\/+$/, "");
  base.pathname = pathname.endsWith("/api") ? pathname : `${pathname}/api`;
  const apiBase = base.href.replace(/\/$/, "");

  function http(path: string) {
    return `${apiBase}/${path.replace(/^\/+/, "").replace(/^api(?:\/|$)/, "")}`;
  }

  function websocket(path = "/ws/predict") {
    const url = new URL(http(path));
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.href;
  }

  function media(path: string | null): string | null {
    if (!path) return null;
    // URLs externas já completas devem conservar sua origem (por exemplo CDN).
    if (/^https?:\/\//i.test(path) || path.startsWith("//")) {
      return new URL(path, base).href;
    }
    // A API retorna /api/media/...; resolve também media/... sem repetir /api.
    if (path.startsWith("/api/") || path.startsWith("api/")) {
      return http(path);
    }
    return path.startsWith("/") ? new URL(path, base.origin).href : http(path);
  }

  return { http, websocket, media };
}
