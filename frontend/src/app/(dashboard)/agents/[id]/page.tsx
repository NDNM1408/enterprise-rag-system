"use client";

import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { useAgent, useDeleteAgent } from "@/lib/hooks";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { KbSelector } from "@/components/agent/kb-selector";
import { ArrowLeft, Loader2, Trash2, Bot, MessageSquare, Settings } from "lucide-react";

export default function AgentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const agentId = params.id as string;

  const { data: agent, isLoading } = useAgent(agentId);
  const deleteAgent = useDeleteAgent();

  const handleDelete = () => {
    if (confirm("Are you sure you want to delete this agent?")) {
      deleteAgent.mutate(agentId, {
        onSuccess: () => {
          router.push("/agents");
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

  if (!agent) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold">Agent not found</h2>
        <Button asChild className="mt-4">
          <Link href="/agents">Back to Agents</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" asChild>
          <Link href="/agents">
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Bot className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">{agent.name}</h1>
              <p className="text-muted-foreground">{agent.description || "No description"}</p>
            </div>
          </div>
        </div>
        <Button asChild>
          <Link href={`/chat/${agent.id}`}>
            <MessageSquare className="mr-2 h-4 w-4" />
            Chat
          </Link>
        </Button>
        <Button variant="destructive" onClick={handleDelete} disabled={deleteAgent.isPending}>
          <Trash2 className="mr-2 h-4 w-4" />
          Delete
        </Button>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <span className="text-sm text-muted-foreground">LLM Model</span>
              <p>
                <Badge variant="secondary">{agent.llm_model}</Badge>
              </p>
            </div>
            <div>
              <span className="text-sm text-muted-foreground">Temperature</span>
              <p className="font-medium">{agent.llm_temperature}</p>
            </div>
            {agent.system_prompt && (
              <div>
                <span className="text-sm text-muted-foreground">System Prompt</span>
                <p className="text-sm mt-1 p-3 bg-muted rounded-lg whitespace-pre-wrap">
                  {agent.system_prompt}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Knowledge Bases</CardTitle>
            <CardDescription>
              Link knowledge bases to provide context for this agent
            </CardDescription>
          </CardHeader>
          <CardContent>
            <KbSelector agent={agent} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
