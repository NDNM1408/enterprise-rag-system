"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useDocuments, useDeleteDocument } from "@/lib/hooks";
import type {
  Document,
  IngestingStatus,
  ParsingStatus,
} from "@/types";
import {
  FileText,
  Loader2,
  Trash2,
  Clock,
  CheckCircle,
  XCircle,
  FileSearch,
  SkipForward,
} from "lucide-react";

interface DocumentListProps {
  kbId: string;
}

const overallStatusConfig = {
  Created: {
    icon: Clock,
    color: "bg-slate-500/10 text-slate-600 border-slate-500/20",
    label: "Queued",
  },
  Processing: {
    icon: Loader2,
    color: "bg-yellow-500/10 text-yellow-600 border-yellow-500/20",
    label: "Processing",
  },
  Succeed: {
    icon: CheckCircle,
    color: "bg-green-500/10 text-green-600 border-green-500/20",
    label: "Done",
  },
  Failed: {
    icon: XCircle,
    color: "bg-red-500/10 text-red-600 border-red-500/20",
    label: "Failed",
  },
} as const;

function isParsingActive(status?: ParsingStatus): boolean {
  return status === "Pending" || status === "Parsing";
}

function isIngestingActive(status?: IngestingStatus): boolean {
  return status === "Pending" || status === "Processing";
}

function PhaseRow({
  label,
  status,
  progress,
  active,
  icon: Icon,
}: {
  label: string;
  status: string;
  progress: number;
  active: boolean;
  icon: React.ComponentType<{ className?: string }>;
}) {
  const isFailed = status === "Failed";
  const isDone = status === "Parsed" || status === "Succeed" || status === "Skipped";
  const indicatorClass = isFailed
    ? "bg-red-500"
    : isDone
      ? "bg-green-500"
      : active
        ? "bg-yellow-500"
        : "bg-muted-foreground/40";

  return (
    <div className="flex items-center gap-2 text-xs">
      <Icon
        className={`h-3 w-3 shrink-0 ${active ? "animate-pulse text-yellow-600" : "text-muted-foreground"}`}
      />
      <span className="w-16 text-muted-foreground">{label}</span>
      <Progress
        value={progress}
        indicatorClassName={indicatorClass}
        className="flex-1"
      />
      <span className="w-10 text-right tabular-nums text-muted-foreground">
        {status === "Skipped" ? "—" : `${Math.round(progress)}%`}
      </span>
    </div>
  );
}

function DocumentStatusCell({ doc }: { doc: Document }) {
  const overall = overallStatusConfig[doc.status] ?? overallStatusConfig.Processing;
  const OverallIcon = overall.icon;
  const showBars =
    isParsingActive(doc.parsing_status) ||
    isIngestingActive(doc.ingesting_status) ||
    doc.status === "Processing";

  return (
    <div className="flex flex-col gap-2 min-w-[220px]">
      <Badge variant="outline" className={overall.color}>
        <OverallIcon
          className={`h-3 w-3 mr-1 ${doc.status === "Processing" ? "animate-spin" : ""}`}
        />
        {overall.label}
      </Badge>
      {showBars && (
        <div className="flex flex-col gap-1">
          <PhaseRow
            label="Parse"
            status={doc.parsing_status ?? "Pending"}
            progress={doc.parsing_progress ?? 0}
            active={isParsingActive(doc.parsing_status)}
            icon={doc.parsing_status === "Skipped" ? SkipForward : FileSearch}
          />
          <PhaseRow
            label="Ingest"
            status={doc.ingesting_status ?? "Pending"}
            progress={doc.ingesting_progress ?? 0}
            active={isIngestingActive(doc.ingesting_status)}
            icon={Loader2}
          />
        </div>
      )}
      {doc.parsing_error && doc.status === "Failed" && (
        <p className="text-xs text-red-600 line-clamp-2" title={doc.parsing_error}>
          {doc.parsing_error}
        </p>
      )}
    </div>
  );
}

export function DocumentList({ kbId }: DocumentListProps) {
  // NOTE: `useDocuments` must be configured with `refetchInterval: 2500` (or
  // similar) while any document is still in a non-terminal phase so the
  // progress bars below tick forward without a page refresh. With React
  // Query, enable conditional polling via:
  //   refetchInterval: (q) => q.state.data?.some(isDocumentActive) ? 2500 : false
  const { data: documents, isLoading } = useDocuments(kbId);
  const deleteDocument = useDeleteDocument();

  const handleDelete = (docId: string, docName: string) => {
    if (confirm(`Are you sure you want to delete "${docName}"?`)) {
      deleteDocument.mutate({ kbId, docId });
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-48">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!documents || documents.length === 0) {
    return (
      <div className="text-center py-12 border rounded-lg">
        <div className="h-12 w-12 rounded-full bg-muted mx-auto flex items-center justify-center mb-4">
          <FileText className="h-6 w-6 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-medium">No documents</h3>
        <p className="text-muted-foreground mt-1">
          Upload documents to populate this knowledge base
        </p>
      </div>
    );
  }

  return (
    <div className="border rounded-lg">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Created</TableHead>
            <TableHead className="w-[50px]"></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {documents.map((doc) => (
            <TableRow key={doc.id}>
              <TableCell>
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">{doc.name}</span>
                </div>
              </TableCell>
              <TableCell>
                <DocumentStatusCell doc={doc} />
              </TableCell>
              <TableCell className="text-muted-foreground">
                {new Date(doc.create_time).toLocaleDateString()}
              </TableCell>
              <TableCell>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => handleDelete(doc.id, doc.name)}
                  disabled={deleteDocument.isPending}
                >
                  <Trash2 className="h-4 w-4 text-destructive" />
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export function isDocumentActive(doc: Document): boolean {
  return (
    doc.status === "Created" ||
    doc.status === "Processing" ||
    isParsingActive(doc.parsing_status) ||
    isIngestingActive(doc.ingesting_status)
  );
}
