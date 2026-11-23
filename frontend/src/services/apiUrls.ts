import { createApiUrls } from "./urls";

export const apiUrls = createApiUrls(import.meta.env.VITE_API_URL, window.location.href);
