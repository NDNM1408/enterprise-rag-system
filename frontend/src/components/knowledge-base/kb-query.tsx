"use client";

import { useState } from "react";
import {
  useQueryKnowledgeBase,
  useDocuments,
  type RetrievedChunk,
  type SearchType,
} from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import {
  Loader2,
  Search,
  ChevronDown,
  ChevronRight,
  Sparkles,
  FileText,
  ChevronLeft,
  Hash,
  Layers,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface KbQueryProps {
  kbId: string;
}

const SEARCH_TYPES: { value: SearchType; label: string; hint: string }[] = [
  { value: "semantic", label: "Semantic", hint: "vector similarity" },
  { value: "hybrid", label: "Hybrid", hint: "vector + full-text" },
  { value: "fuzzy", label: "Fuzzy", hint: "trigram / full-text" },
];

export function KbQuery({ kbId }: KbQueryProps) {
  const [queryText, setQueryText] = useState("");
  const [searchType, setSearchType] = useState<SearchType>("semantic");
  const [topK, setTopK] = useState(5);
  const query = useQueryKnowledgeBase();
  const { data: documents } = useDocuments(kbId);

  const docNameById = (id: string) =>
    documents?.find((d) => d.id === id)?.name ?? id.slice(0, 8) + "…";

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = queryText.trim();
    if (!text || query.isPending) return;
    query.mutate({ kbId, queryText: text, topK, searchType });
  };

  return (
    <div className="space-y-6">
      {/* Search bar */}
      <Card className="border-primary/10">
        <CardContent className="p-4 space-y-3">
          <form onSubmit={handleSubmit} className="space-y-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Try a query — a phrase, a question, a keyword…"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                disabled={query.isPending}
                className="pl-10 pr-32 h-11 text-base"
              />
              <Button
                type="submit"
                size="sm"
                disabled={query.isPending || !queryText.trim()}
                className="absolute right-1.5 top-1/2 -translate-y-1/2 h-8"
              >
                {query.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <>
                    <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                    Retrieve
                  </>
                )}
              </Button>
            </div>

            <div className="flex flex-wrap items-center gap-4 text-sm">
              {/* Search-type pill toggle */}
              <div className="flex items-center gap-1 rounded-full border bg-muted/40 p-0.5">
                {SEARCH_TYPES.map((t) => (
                  <button
                    key={t.value}
                    type="button"
                    onClick={() => setSearchType(t.value)}
                    className={cn(
                      "px-3 py-1 text-xs font-medium rounded-full transition-colors",
                      searchType === t.value
                        ? "bg-background shadow-sm text-foreground"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                    title={t.hint}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              {/* Top-K stepper */}
              <div className="flex items-center gap-2 ml-auto">
                <span className="text-xs text-muted-foreground">Top</span>
                <div className="flex items-center rounded-md border">
                  <button
                    type="button"
                    className="px-2 py-1 hover:bg-muted text-muted-foreground"
                    onClick={() => setTopK((k) => Math.max(1, k - 1))}
                  >
                    <ChevronLeft className="h-3 w-3" />
                  </button>
                  <span className="px-2 text-sm font-mono w-8 text-center">{topK}</span>
                  <button
                    type="button"
                    className="px-2 py-1 hover:bg-muted text-muted-foreground"
                    onClick={() => setTopK((k) => Math.min(50, k + 1))}
                  >
                    <ChevronRight className="h-3 w-3" />
                  </button>
                </div>
              </div>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Error */}
      {query.error && (
        <Card className="border-destructive/30 bg-destructive/5">
          <CardContent className="p-3 text-sm text-destructive flex items-start gap-2">
            <span className="font-semibold">Error:</span>
            <span>{query.error.message}</span>
          </CardContent>
        </Card>
      )}

      {/* Empty state — before first query */}
      {!query.data && !query.isPending && !query.error && (
        <div className="text-center py-16 text-muted-foreground">
          <div className="mx-auto mb-3 h-12 w-12 rounded-full bg-muted/40 flex items-center justify-center">
            <Search className="h-6 w-6" />
          </div>
          <p className="text-sm">Run a query above to see retrieved chunks.</p>
          <p className="text-xs mt-1 opacity-70">
            Same retrieval the agent uses — handy for debugging chunking + ranking.
          </p>
        </div>
      )}

      {/* Results */}
      {query.data && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <div className="text-muted-foreground">
              <span className="font-semibold text-foreground">
                {query.data.result_count}
              </span>{" "}
              result{query.data.result_count === 1 ? "" : "s"}
              {" "}for{" "}
              <span className="font-mono text-foreground">
                &quot;{query.data.query_text}&quot;
              </span>
            </div>
            <Badge variant="outline" className="text-xs">
              {query.data.search_type}
            </Badge>
          </div>
          <Separator />
          {query.data.results.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground text-sm">
              No chunks matched. Try a different query or switch to{" "}
              <button
                type="button"
                className="text-primary hover:underline"
                onClick={() => setSearchType(searchType === "fuzzy" ? "semantic" : "fuzzy")}
              >
                {searchType === "fuzzy" ? "semantic" : "fuzzy"}
              </button>
              {" "}search.
            </div>
          ) : (
            <div className="space-y-3">
              {query.data.results.map((chunk, i) => (
                <ChunkCard
                  key={chunk.chunk_id}
                  rank={i + 1}
                  chunk={chunk}
                  documentName={docNameById(chunk.document_id)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface ChunkCardProps {
  rank: number;
  chunk: RetrievedChunk;
  documentName: string;
}

function ChunkCard({ rank, chunk, documentName }: ChunkCardProps) {
  const [showParent, setShowParent] = useState(false);
  const score = chunk.similarity ?? chunk.score;
  const heading = chunk.metadata?.heading_path;
  const tokens = chunk.metadata?.tokens;
  const hasParent =
    chunk.parent_text &&
    chunk.parent_text.length > 0 &&
    chunk.parent_text !== chunk.text;

  // ``score`` is cosine similarity (0..1) for semantic / fuzzy; for hybrid it
  // is a normalized rank score. Clamp to [0,1] for the progress bar.
  const scorePct = score !== undefined ? Math.max(0, Math.min(1, score)) * 100 : null;

  return (
    <Card className="group transition-colors hover:border-primary/30">
      <CardContent className="p-4 space-y-3">
        {/* Header row */}
        <div className="flex items-center gap-3">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary text-xs font-semibold">
            {rank}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 text-sm text-muted-foreground truncate">
              <FileText className="h-3.5 w-3.5 shrink-0" />
              <span className="truncate">{documentName}</span>
              {heading && (
                <>
                  <ChevronRight className="h-3 w-3 shrink-0 opacity-50" />
                  <span className="truncate font-medium text-foreground">
                    {heading}
                  </span>
                </>
              )}
            </div>
          </div>
          {tokens !== undefined && (
            <Badge variant="outline" className="text-[10px] font-mono">
              <Hash className="h-2.5 w-2.5 mr-0.5" />
              {tokens}
            </Badge>
          )}
        </div>

        {/* Score */}
        {scorePct !== null && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground w-12 shrink-0">score</span>
            <Progress value={scorePct} className="h-1.5 flex-1" />
            <span className="font-mono w-14 text-right">
              {score?.toFixed(4)}
            </span>
          </div>
        )}

        {/* Body — embed text */}
        <pre className="whitespace-pre-wrap break-words text-sm leading-relaxed bg-muted/40 rounded-md p-3 max-h-72 overflow-auto font-mono text-[12.5px]">
          {chunk.text}
        </pre>

        {/* Parent toggle */}
        {hasParent && (
          <div>
            <button
              type="button"
              onClick={() => setShowParent((s) => !s)}
              className="flex items-center gap-1 text-xs text-primary hover:underline"
            >
              {showParent ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
              <Layers className="h-3 w-3" />
              {showParent ? "Hide parent_text" : "Show parent_text (LLM context)"}
            </button>
            {showParent && (
              <pre className="mt-2 whitespace-pre-wrap break-words text-sm leading-relaxed bg-muted/30 border-l-2 border-primary/30 rounded-md p-3 max-h-96 overflow-auto font-mono text-[12.5px]">
                {chunk.parent_text}
              </pre>
            )}
          </div>
        )}

        {/* Footer IDs */}
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground font-mono opacity-0 group-hover:opacity-100 transition-opacity">
          <span>chunk {chunk.chunk_id.slice(0, 8)}</span>
          <span>·</span>
          <span>doc {chunk.document_id.slice(0, 8)}</span>
        </div>
      </CardContent>
    </Card>
  );
}
