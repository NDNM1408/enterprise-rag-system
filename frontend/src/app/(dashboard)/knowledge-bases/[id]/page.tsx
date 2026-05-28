"use client";

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

export default function KnowledgeBaseDetailPage() {
  const params = useParams();
  const router = useRouter();
  const kbId = params.id as string;

  const { data: kb, isLoading } = useKnowledgeBase(kbId);
  const deleteKb = useDeleteKnowledgeBase();
  const updateKb = useUpdateKnowledgeBase();

  const handleToggleAgentic = (next: boolean) => {
    if (!kb) return;
    updateKb.mutate({
      kbId,
      patch: {
        parser_config: {
          rag_mode: kb.parser_config?.rag_mode || "classic",
          agentic_search: next,
        },
      },
    });
  };

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
                Planner LLM fans out across pivot axes per query (up to 5
                iters). Higher recall on cross-basin queries; slower.
              </p>
            </div>
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
