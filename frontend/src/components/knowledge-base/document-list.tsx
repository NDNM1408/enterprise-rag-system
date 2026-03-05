"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useDocuments, useDeleteDocument } from "@/lib/hooks";
import { FileText, Loader2, Trash2, Clock, CheckCircle, XCircle } from "lucide-react";

interface DocumentListProps {
  kbId: string;
}

const statusConfig = {
  Processing: {
    icon: Clock,
    color: "bg-yellow-500/10 text-yellow-600 border-yellow-500/20",
  },
  Succeed: {
    icon: CheckCircle,
    color: "bg-green-500/10 text-green-600 border-green-500/20",
  },
  Failed: {
    icon: XCircle,
    color: "bg-red-500/10 text-red-600 border-red-500/20",
  },
};

export function DocumentList({ kbId }: DocumentListProps) {
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
          {documents.map((doc) => {
            const status = statusConfig[doc.status];
            const StatusIcon = status.icon;

            return (
              <TableRow key={doc.id}>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <span className="font-medium">{doc.name}</span>
                  </div>
                </TableCell>
                <TableCell>
                  <Badge variant="outline" className={status.color}>
                    <StatusIcon className="h-3 w-3 mr-1" />
                    {doc.status}
                  </Badge>
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
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
