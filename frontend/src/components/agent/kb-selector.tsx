"use client";

import { useKnowledgeBases, useLinkKnowledgeBase, useUnlinkKnowledgeBase } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, Plus, X, Database } from "lucide-react";
import type { Agent } from "@/types";

interface KbSelectorProps {
  agent: Agent;
}

export function KbSelector({ agent }: KbSelectorProps) {
  const { data: knowledgeBases, isLoading } = useKnowledgeBases();
  const linkKb = useLinkKnowledgeBase();
  const unlinkKb = useUnlinkKnowledgeBase();

  const linkedKbIds = new Set(agent.knowledge_bases?.map((kb) => kb.id) || []);
  const availableKbs = knowledgeBases?.filter((kb) => !linkedKbIds.has(kb.id)) || [];

  const handleLink = (kbId: string) => {
    linkKb.mutate({ agentId: agent.id, kbId });
  };

  const handleUnlink = (kbId: string) => {
    unlinkKb.mutate({ agentId: agent.id, kbId });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-4">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-medium mb-3">Linked Knowledge Bases</h3>
        {agent.knowledge_bases && agent.knowledge_bases.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {agent.knowledge_bases.map((kb) => (
              <Badge
                key={kb.id}
                variant="secondary"
                className="flex items-center gap-1 py-1.5 px-3"
              >
                <Database className="h-3 w-3 mr-1" />
                {kb.name}
                <button
                  onClick={() => handleUnlink(kb.id)}
                  className="ml-1 hover:text-destructive"
                  disabled={unlinkKb.isPending}
                >
                  <X className="h-3 w-3" />
                </button>
              </Badge>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            No knowledge bases linked yet
          </p>
        )}
      </div>

      {availableKbs.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-3">Available Knowledge Bases</h3>
          <div className="flex flex-wrap gap-2">
            {availableKbs.map((kb) => (
              <Button
                key={kb.id}
                variant="outline"
                size="sm"
                onClick={() => handleLink(kb.id)}
                disabled={linkKb.isPending}
              >
                <Plus className="h-3 w-3 mr-1" />
                {kb.name}
              </Button>
            ))}
          </div>
        </div>
      )}

      {knowledgeBases && knowledgeBases.length === 0 && (
        <p className="text-sm text-muted-foreground">
          No knowledge bases available. Create one first to link it to this agent.
        </p>
      )}
    </div>
  );
}
