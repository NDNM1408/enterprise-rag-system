"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  useKnowledgeBase,
  useDeleteKnowledgeBase,
  useUpdateKnowledgeBase,
} from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DocumentUpload } from "@/components/knowledge-base/document-upload";
import { DocumentList } from "@/components/knowledge-base/document-list";
import { KbQuery } from "@/components/knowledge-base/kb-query";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ArrowLeft, Loader2, Trash2, Database, FileText, Search, Sparkles } from "lucide-react";

/** Numeric input that PATCHes on blur. Empty value clears the override
 *  (server then falls back to the global default placeholder). */
function KbNumberKnob({
  label,
  hint,
  value,
  placeholder,
  disabled,
  onCommit,
}: {
  label: string;
  hint: string;
  value: number | null | undefined;
  placeholder: number;
  disabled?: boolean;
  onCommit: (v: number | null) => void;
}) {
  const [local, setLocal] = useState<string>(value != null ? String(value) : "");
  useEffect(() => {
    setLocal(value != null ? String(value) : "");
  }, [value]);
  const commit = () => {
    const trimmed = local.trim();
    if (trimmed === "") {
      onCommit(null);
      return;
    }
    const n = Number(trimmed);
    if (Number.isFinite(n) && n > 0) {
      onCommit(Math.floor(n));
    } else {
      setLocal(value != null ? String(value) : "");
    }
  };
  return (
    <div className="flex flex-col">
      <label className="text-xs text-muted-foreground mb-1">{label}</label>
      <input
        type="number"
        min={1}
        max={50}
        disabled={disabled}
        value={local}
        placeholder={`default ${placeholder}`}
        onChange={(e) => setLocal(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
        className="w-32 rounded-md border border-input bg-background px-2 py-1 text-sm shadow-xs disabled:opacity-50"
      />
      <span className="text-[10px] text-muted-foreground mt-0.5 max-w-[10rem]">
        {hint}
      </span>
    </div>
  );
}

export default function KnowledgeBaseDetailPage() {
  const params = useParams();
  const router = useRouter();
  const kbId = params.id as string;

  const { data: kb, isLoading } = useKnowledgeBase(kbId);
  const deleteKb = useDeleteKnowledgeBase();
  const updateKb = useUpdateKnowledgeBase();

  // Build a parser_config patch that preserves the current state and
  // overrides only the fields we're changing — server replaces the whole
  // jsonb wholesale, so we must always send the complete config.
  const patchConfig = (overrides: Partial<NonNullable<typeof kb>["parser_config"]>) => {
    if (!kb) return;
    updateKb.mutate({
      kbId,
      patch: {
        parser_config: {
          rag_mode: kb.parser_config?.rag_mode || "classic",
          agentic_search: kb.parser_config?.agentic_search || false,
          top_n: kb.parser_config?.top_n,
          agentic_max_iter: kb.parser_config?.agentic_max_iter,
          agentic_top_k_per_iter: kb.parser_config?.agentic_top_k_per_iter,
          ...overrides,
        },
      },
    });
  };

  const handleToggleAgentic = (next: boolean) => patchConfig({ agentic_search: next });

  const handleDelete = () => {
    if (confirm("Are you sure you want to delete this knowledge base?")) {
      deleteKb.mutate(kbId, {
        onSuccess: () => {
          router.push("/knowledge-bases");
        },
      });
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!kb) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold">Knowledge base not found</h2>
        <Button asChild className="mt-4">
          <Link href="/knowledge-bases">Back to Knowledge Bases</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" asChild>
          <Link href="/knowledge-bases">
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Database className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">{kb.name}</h1>
              <p className="text-muted-foreground">{kb.description || "No description"}</p>
            </div>
          </div>
        </div>
        <Button variant="destructive" onClick={handleDelete} disabled={deleteKb.isPending}>
          <Trash2 className="mr-2 h-4 w-4" />
          Delete
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-start gap-6">
            <div>
              <span className="text-sm text-muted-foreground">RAG Mode</span>
              <p>
                <Badge variant="secondary">
                  {kb.parser_config?.rag_mode || "classic"}
                </Badge>
              </p>
            </div>
            <div className="flex flex-col">
              <span className="text-sm text-muted-foreground mb-1">
                Agentic search
              </span>
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-input"
                  checked={!!kb.parser_config?.agentic_search}
                  disabled={updateKb.isPending}
                  onChange={(e) => handleToggleAgentic(e.target.checked)}
                />
                <Badge
                  variant={
                    kb.parser_config?.agentic_search ? "default" : "secondary"
                  }
                  className="gap-1"
                >
                  <Sparkles className="h-3 w-3" />
                  {kb.parser_config?.agentic_search ? "ON" : "OFF"}
                </Badge>
                {updateKb.isPending && (
                  <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                )}
              </label>
              <p className="text-xs text-muted-foreground mt-1 max-w-md">
                Planner LLM fans out across pivot axes per query. Higher
                recall on cross-basin queries; slower.
              </p>
            </div>
            <div className="basis-full" />
            <KbNumberKnob
              label="Top N (returned chunks)"
              hint="Final chunks fed to the answer LLM."
              value={kb.parser_config?.top_n}
              placeholder={10}
              onCommit={(v) => patchConfig({ top_n: v as number | undefined })}
            />
            <KbNumberKnob
              label="Max iterations"
              hint="Hard cap on planner hops (agentic only)."
              value={kb.parser_config?.agentic_max_iter}
              placeholder={5}
              disabled={!kb.parser_config?.agentic_search}
              onCommit={(v) =>
                patchConfig({ agentic_max_iter: v as number | undefined })
              }
            />
            <KbNumberKnob
              label="Top K per iter"
              hint="Vector top_k per sub-query per hop."
              value={kb.parser_config?.agentic_top_k_per_iter}
              placeholder={5}
              disabled={!kb.parser_config?.agentic_search}
              onCommit={(v) =>
                patchConfig({
                  agentic_top_k_per_iter: v as number | undefined,
                })
              }
            />
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="documents" className="space-y-4">
        <TabsList>
          <TabsTrigger value="documents" className="gap-2">
            <FileText className="h-4 w-4" />
            Documents
          </TabsTrigger>
          <TabsTrigger value="query" className="gap-2">
            <Search className="h-4 w-4" />
            Try a query
          </TabsTrigger>
        </TabsList>
        <TabsContent value="documents" className="space-y-4">
          <DocumentUpload kbId={kbId} />
          <DocumentList kbId={kbId} />
        </TabsContent>
        <TabsContent value="query" className="space-y-4">
          <KbQuery kbId={kbId} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
