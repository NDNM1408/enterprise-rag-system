"use client";

import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { api } from "./api";
import type {
  CreateKbRequest,
  Document,
  KnowledgeBase,
} from "@/types";

// ---------------------------------------------------------------------------
//  Documents
// ---------------------------------------------------------------------------

function isDocumentActive(doc: Document): boolean {
  return (
    doc.status === "Created" ||
    doc.status === "Processing" ||
    doc.parsing_status === "Pending" ||
    doc.parsing_status === "Parsing" ||
    doc.ingesting_status === "Pending" ||
    doc.ingesting_status === "Processing"
  );
}

interface ListDocumentsResponse {
  kb_id: string;
  count: number;
  documents: Document[];
}

export function useDocuments(kbId: string) {
  return useQuery<Document[]>({
    queryKey: ["documents", kbId],
    queryFn: async () => {
      const { data } = await api.get<ListDocumentsResponse>(
        `/api/v1/${kbId}/documents`,
      );
      return data.documents ?? [];
    },
    // Poll while any document is mid-pipeline. As soon as everything is
    // terminal (Succeed/Failed) the polling stops.
    refetchInterval: (query) =>
      query.state.data?.some(isDocumentActive) ? 2500 : false,
  });
}

export function useUploadDocuments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ kbId, files }: { kbId: string; files: File[] }) => {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      const { data } = await api.post(`/api/v1/${kbId}/documents`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return data;
    },
    onSuccess: (_data, { kbId }) => {
      qc.invalidateQueries({ queryKey: ["documents", kbId] });
    },
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async ({ kbId, docId }: { kbId: string; docId: string }) => {
      const { data } = await api.delete(`/api/v1/${kbId}/documents/${docId}`);
      return data;
    },
    onSuccess: (_data, { kbId }) => {
      qc.invalidateQueries({ queryKey: ["documents", kbId] });
    },
  });
}

// ---------------------------------------------------------------------------
//  Knowledge bases
// ---------------------------------------------------------------------------

interface PagedKbResponse {
  total: number;
  page: number;
  page_size: number;
  items: KnowledgeBase[];
}

export function useKnowledgeBases() {
  return useQuery<KnowledgeBase[]>({
    queryKey: ["knowledge-bases"],
    queryFn: async () => {
      const { data } = await api.get<PagedKbResponse>("/api/v1/knowledge_base/", {
        params: { page: 1, page_size: 100 },
      });
      return data.items ?? [];
    },
  });
}

export function useKnowledgeBase(kbId: string) {
  return useQuery<KnowledgeBase>({
    queryKey: ["knowledge-base", kbId],
    queryFn: async () => {
      const { data } = await api.get(`/api/v1/knowledge_base/${kbId}`);
      return data;
    },
    enabled: !!kbId,
  });
}

export function useCreateKnowledgeBase() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (body: CreateKbRequest) => {
      const { data } = await api.post("/api/v1/knowledge_base", body);
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["knowledge-bases"] });
    },
  });
}

export function useDeleteKnowledgeBase() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (kbId: string) => {
      const { data } = await api.delete(`/api/v1/knowledge_base/${kbId}`);
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["knowledge-bases"] });
    },
  });
}
