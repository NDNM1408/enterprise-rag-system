"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import { useAgent, useChat } from "@/lib/hooks";
import { useChatStore } from "@/lib/stores";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChatContainer } from "@/components/chat";
import { ArrowLeft, Loader2, Bot, Plus } from "lucide-react";
import { useEffect } from "react";

export default function ChatWithAgentPage() {
  const params = useParams();
  const agentId = params.agentId as string;

  const { data: agent, isLoading } = useAgent(agentId);
  const chat = useChat(agentId);
  const { reset, isStreaming } = useChatStore();

  // Reset chat state when leaving the page
  useEffect(() => {
    return () => {
      reset();
    };
  }, [reset]);

  const handleSendMessage = (message: string) => {
    chat.mutate(message);
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
          <Link href="/chat">Back to Chat</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="flex gap-6 h-[calc(100vh-8rem)]">
      {/* Main chat area */}
      <div className="flex-1 flex flex-col">
        <div className="flex items-center gap-4 mb-4">
          <Button variant="ghost" size="icon" asChild>
            <Link href="/chat">
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <Bot className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold">{agent.name}</h1>
              <p className="text-sm text-muted-foreground">
                {agent.knowledge_bases?.length || 0} knowledge bases linked
              </p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => reset()}
            className="ml-auto"
          >
            <Plus className="h-4 w-4 mr-2" />
            New Chat
          </Button>
        </div>

        <Card className="flex-1 flex flex-col overflow-hidden">
          <CardContent className="flex-1 flex flex-col p-0 overflow-hidden">
            <ChatContainer
              onSendMessage={handleSendMessage}
              isLoading={isStreaming || chat.isPending}
            />
          </CardContent>
        </Card>
      </div>

      {/* Sidebar - Agent info */}
      <div className="w-80 hidden lg:block">
        <Card className="h-full flex flex-col">
          <CardHeader>
            <CardTitle className="text-lg">Agent Info</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 flex-1 min-h-0 overflow-y-auto">
            <div>
              <span className="text-sm text-muted-foreground">Model</span>
              <p className="font-medium text-sm">{agent.llm_model}</p>
            </div>
            <div>
              <span className="text-sm text-muted-foreground">Temperature</span>
              <p className="font-medium text-sm">{agent.llm_temperature}</p>
            </div>
            {agent.knowledge_bases && agent.knowledge_bases.length > 0 && (
              <div>
                <span className="text-sm text-muted-foreground">
                  Knowledge Bases ({agent.knowledge_bases.length})
                </span>
                <div className="flex flex-wrap gap-2 mt-2">
                  {agent.knowledge_bases.map((kb) => (
                    <span
                      key={kb.id}
                      className="text-xs px-2 py-1 bg-muted rounded-md"
                    >
                      {kb.name}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {agent.system_prompt && (
              <div>
                <span className="text-sm text-muted-foreground">System Prompt</span>
                <p className="text-xs mt-1 p-2 bg-muted rounded-lg whitespace-pre-wrap break-words">
                  {agent.system_prompt}
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
