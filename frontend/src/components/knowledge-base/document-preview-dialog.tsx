"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useDocumentPreview } from "@/lib/hooks";
import { cn } from "@/lib/utils";
import {
  FileText,
  Layers,
  Loader2,
  ChevronRight,
  ChevronDown,
  Code2,
  Image as ImageIcon,
} from "lucide-react";

interface DocumentPreviewDialogProps {
  kbId: string;
  docId: string | null;
  docName?: string;
  onOpenChange: (open: boolean) => void;
}

// Tailwind utility group that approximates `prose` without pulling in the
// typography plugin. Applied to the markdown-rendered output.
const markdownClasses = cn(
  "max-w-none text-sm leading-relaxed",
  "[&_h1]:text-2xl [&_h1]:font-bold [&_h1]:mt-6 [&_h1]:mb-3",
  "[&_h2]:text-xl [&_h2]:font-semibold [&_h2]:mt-5 [&_h2]:mb-2",
  "[&_h3]:text-lg [&_h3]:font-semibold [&_h3]:mt-4 [&_h3]:mb-2",
  "[&_h4]:text-base [&_h4]:font-semibold [&_h4]:mt-3 [&_h4]:mb-2",
  "[&_p]:my-2",
  "[&_ul]:list-disc [&_ul]:pl-6 [&_ul]:my-2",
  "[&_ol]:list-decimal [&_ol]:pl-6 [&_ol]:my-2",
  "[&_li]:my-1",
  "[&_a]:text-primary [&_a]:underline",
  "[&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs [&_code]:font-mono",
  "[&_pre]:rounded [&_pre]:bg-muted [&_pre]:p-3 [&_pre]:my-3 [&_pre]:overflow-x-auto",
  "[&_pre_code]:bg-transparent [&_pre_code]:p-0",
  "[&_blockquote]:border-l-4 [&_blockquote]:border-muted [&_blockquote]:pl-4 [&_blockquote]:italic [&_blockquote]:text-muted-foreground",
  "[&_hr]:my-4 [&_hr]:border-border",
  // Tables (GFM and inline HTML alike)
  "[&_table]:my-4 [&_table]:w-full [&_table]:border-collapse [&_table]:text-xs",
  "[&_th]:border [&_th]:border-border [&_th]:bg-muted/50 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_th]:font-medium",
  "[&_td]:border [&_td]:border-border [&_td]:px-2 [&_td]:py-1 [&_td]:align-top",
);

function MarkdownView({ source }: { source: string }) {
  return (
    <div className={markdownClasses}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          // MinerU output references images with relative paths (e.g.
          // `images/page_007_image_04.jpg`) that aren't served by the UI.
          // Render a small inline placeholder instead of a broken icon.
          img: ({ src, alt }) => (
            <span className="inline-flex items-center gap-1 rounded border border-dashed border-muted-foreground/40 bg-muted/30 px-2 py-1 text-xs text-muted-foreground">
              <ImageIcon className="h-3 w-3" />
              <span className="font-mono truncate max-w-[280px]">
                {alt || (typeof src === "string" ? src.split("/").pop() : "image")}
              </span>
            </span>
          ),
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}

export function DocumentPreviewDialog({
  kbId,
  docId,
  docName,
  onOpenChange,
}: DocumentPreviewDialogProps) {
  const open = !!docId;
  const [showRaw, setShowRaw] = useState(false);
  const { data, isLoading, isError, error } = useDocumentPreview(kbId, docId);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className={cn(
          "w-[95vw] max-w-[1400px] sm:max-w-[1400px] h-[92vh] p-0 gap-0 flex flex-col",
        )}
      >
        <DialogHeader className="px-6 pt-6 pb-3 border-b">
          <DialogTitle className="flex items-center gap-2 pr-10">
            <FileText className="h-5 w-5 text-muted-foreground shrink-0" />
            <span className="truncate">
              {data?.name || docName || "Document preview"}
            </span>
          </DialogTitle>
          <DialogDescription asChild>
            {data ? (
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <Badge variant="secondary">{data.status}</Badge>
                {data.parsing_status && (
                  <Badge variant="outline">parse: {data.parsing_status}</Badge>
                )}
                {data.ingesting_status && (
                  <Badge variant="outline">ingest: {data.ingesting_status}</Badge>
                )}
                <span className="text-muted-foreground">
                  {data.chunk_count} chunk{data.chunk_count === 1 ? "" : "s"}
                </span>
              </div>
            ) : (
              <span>Inspect parsed output and stored chunks</span>
            )}
          </DialogDescription>
        </DialogHeader>

        {isLoading && (
          <div className="flex-1 flex items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
          </div>
        )}

        {isError && (
          <div className="p-8 text-sm text-red-600">
            Failed to load preview: {(error as Error)?.message || "unknown error"}
          </div>
        )}

        {data && (
          <Tabs
            defaultValue="markdown"
            className="flex-1 min-h-0 flex flex-col px-6 pb-6 pt-3"
          >
            <div className="flex items-center justify-between">
              <TabsList>
                <TabsTrigger value="markdown" className="gap-2">
                  <FileText className="h-4 w-4" />
                  {data.parsed_source === "original" ? "Source" : "Parsed"}
                </TabsTrigger>
                <TabsTrigger value="chunks" className="gap-2">
                  <Layers className="h-4 w-4" />
                  Chunks ({data.chunk_count})
                </TabsTrigger>
              </TabsList>
              <button
                type="button"
                onClick={() => setShowRaw((v) => !v)}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                title="Toggle raw source"
              >
                <Code2 className="h-3.5 w-3.5" />
                {showRaw ? "Rendered" : "Raw"}
              </button>
            </div>

            <TabsContent
              value="markdown"
              className="flex-1 min-h-0 mt-3 data-[state=inactive]:hidden"
            >
              {data.parsed_markdown ? (
                <ScrollArea className="h-full rounded-md border bg-background">
                  <div className="p-6">
                    {showRaw ? (
                      <pre className="text-xs whitespace-pre-wrap break-words font-mono leading-relaxed">
                        {data.parsed_markdown}
                      </pre>
                    ) : (
                      <MarkdownView source={data.parsed_markdown} />
                    )}
                  </div>
                </ScrollArea>
              ) : (
                <div className="rounded-md border bg-muted/20 p-6 text-sm text-muted-foreground">
                  No source available for this document.
                </div>
              )}
            </TabsContent>

            <TabsContent
              value="chunks"
              className="flex-1 min-h-0 mt-3 data-[state=inactive]:hidden"
            >
              <ScrollArea className="h-full rounded-md border">
                <div className="p-4 space-y-3">
                  {data.chunks.length === 0 && (
                    <div className="text-sm text-muted-foreground p-3">
                      No chunks were stored for this document.
                    </div>
                  )}
                  {data.chunks.map((c, idx) => (
                    <ChunkCard
                      key={c.id}
                      index={idx}
                      chunk={c}
                      raw={showRaw}
                    />
                  ))}
                </div>
              </ScrollArea>
            </TabsContent>
          </Tabs>
        )}
      </DialogContent>
    </Dialog>
  );
}

function ChunkCard({
  index,
  chunk,
  raw,
}: {
  index: number;
  chunk: { id: string; content: string; parent_text: string; status: string };
  raw: boolean;
}) {
  const [showParent, setShowParent] = useState(false);
  const hasParent = !!chunk.parent_text && chunk.parent_text !== chunk.content;

  return (
    <Card>
      <CardContent className="p-4 space-y-2">
        <div className="flex items-center gap-2 text-xs">
          <Badge variant="outline" className="font-mono">
            #{index + 1}
          </Badge>
          <Badge
            variant="secondary"
            className={
              chunk.status === "Succeed"
                ? "bg-green-500/10 text-green-700"
                : chunk.status === "Failed"
                  ? "bg-red-500/10 text-red-700"
                  : ""
            }
          >
            {chunk.status}
          </Badge>
          <span
            className="font-mono text-muted-foreground truncate"
            title={chunk.id}
          >
            {chunk.id.slice(0, 8)}…
          </span>
        </div>
        {raw ? (
          <pre className="text-xs whitespace-pre-wrap break-words font-mono leading-relaxed">
            {chunk.content}
          </pre>
        ) : (
          <MarkdownView source={chunk.content} />
        )}
        {hasParent && (
          <>
            <Separator />
            <button
              type="button"
              onClick={() => setShowParent((v) => !v)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            >
              {showParent ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
              {showParent ? "Hide" : "Show"} parent section
            </button>
            {showParent && (
              <div className="rounded bg-muted/30 p-3">
                {raw ? (
                  <pre className="text-xs whitespace-pre-wrap break-words font-mono leading-relaxed">
                    {chunk.parent_text}
                  </pre>
                ) : (
                  <MarkdownView source={chunk.parent_text} />
                )}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
