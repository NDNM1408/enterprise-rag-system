"use client";

import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Bot, MoreVertical, Trash2, MessageSquare } from "lucide-react";
import type { Agent } from "@/types";

interface AgentCardProps {
  agent: Agent;
  onDelete: (id: string) => void;
}

export function AgentCard({ agent, onDelete }: AgentCardProps) {
  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="flex flex-row items-start justify-between space-y-0">
        <div className="flex items-start gap-3">
          <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
            <Bot className="h-5 w-5 text-primary" />
          </div>
          <div>
            <Link href={`/agents/${agent.id}`}>
              <CardTitle className="text-lg hover:text-primary cursor-pointer">
                {agent.name}
              </CardTitle>
            </Link>
            <CardDescription className="mt-1 line-clamp-2">
              {agent.description || "No description"}
            </CardDescription>
          </div>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <MoreVertical className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem asChild>
              <Link href={`/chat/${agent.id}`}>
                <MessageSquare className="mr-2 h-4 w-4" />
                Chat
              </Link>
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => onDelete(agent.id)}
              className="text-destructive focus:text-destructive"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-4 text-sm text-muted-foreground">
          <span>{agent.knowledge_bases?.length || 0} knowledge bases</span>
          <Badge variant="secondary">{agent.llm_model}</Badge>
        </div>
      </CardContent>
    </Card>
  );
}
