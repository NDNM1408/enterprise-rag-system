import axios, { AxiosResponse } from "axios";

// data-api (KBs, documents, agents, chat, conversations) — the chatbot
// service has been folded into data-api so all endpoints share one host.
// Override at deploy time with NEXT_PUBLIC_API_URL.
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// LiteLLM proxy (OpenAI-compatible) — used to pull the live model list for
// the agent-creation dropdown so the UI always reflects the gateway's config.
export const LITELLM_BASE_URL =
  process.env.NEXT_PUBLIC_LITELLM_URL || "http://localhost:4000";
export const LITELLM_API_KEY =
  process.env.NEXT_PUBLIC_LITELLM_API_KEY || "fake";

function unwrapEnvelope(response: AxiosResponse): AxiosResponse {
  if (response.data && typeof response.data === "object" && "data" in response.data) {
    return { ...response, data: response.data.data };
  }
  return response;
}

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});
api.interceptors.response.use(unwrapEnvelope);

// LiteLLM returns OpenAI's native shape ({ data: [...], object: "list" }) —
// don't unwrap; consumers read response.data.data directly.
export const litellmApi = axios.create({
  baseURL: LITELLM_BASE_URL,
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${LITELLM_API_KEY}`,
  },
});
