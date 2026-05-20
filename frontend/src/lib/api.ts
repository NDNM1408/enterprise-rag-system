import axios from "axios";

// data-api is reachable through a Next.js rewrite or directly via this base URL.
// Override at deploy time with NEXT_PUBLIC_API_URL.
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});

// data-api wraps every response in { data, request_id, timestamp, ... }.
// Unwrap so consumers see the payload directly.
api.interceptors.response.use((response) => {
  if (response.data && typeof response.data === "object" && "data" in response.data) {
    return { ...response, data: response.data.data };
  }
  return response;
});
