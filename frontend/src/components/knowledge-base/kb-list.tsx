"use client";

import { useState } from "react";
import { KbCard } from "./kb-card";
import { CreateKbDialog } from "./create-kb-dialog";
import { Button } from "@/components/ui/button";
import { useKnowledgeBases, useDeleteKnowledgeBase } from "@/lib/hooks";
import { Plus, Loader2 } from "lucide-react";

export function KbList() {
  const { data: knowledgeBases, isLoading } = useKnowledgeBases();
  const deleteKb = useDeleteKnowledgeBase();
  const [isCreateOpen, setIsCreateOpen] = useState(false);

  const handleDelete = (id: string) => {
    if (confirm("Are you sure you want to delete this knowledge base?")) {
      deleteKb.mutate(id);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Knowledge Bases</h1>
          <p className="text-muted-foreground mt-1">
            Manage your document collections and vector stores
          </p>
        </div>
        <Button onClick={() => setIsCreateOpen(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Create Knowledge Base
        </Button>
      </div>

      {knowledgeBases && knowledgeBases.length > 0 ? (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {knowledgeBases.map((kb) => (
            <KbCard key={kb.id} kb={kb} onDelete={handleDelete} />
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="h-12 w-12 rounded-full bg-muted mx-auto flex items-center justify-center mb-4">
            <Plus className="h-6 w-6 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-medium">No knowledge bases</h3>
          <p className="text-muted-foreground mt-1 mb-4">
            Get started by creating your first knowledge base
          </p>
          <Button onClick={() => setIsCreateOpen(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Create Knowledge Base
          </Button>
        </div>
      )}

      <CreateKbDialog open={isCreateOpen} onOpenChange={setIsCreateOpen} />
    </div>
  );
}
