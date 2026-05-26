"use client";

import { useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageBubble } from "./message-bubble";
import { ChatInput } from "./chat-input";
import { useChatStore } from "@/lib/stores";

interface ChatContainerProps {
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
}

export function ChatContainer({ onSendMessage, isLoading }: ChatContainerProps) {
  const { messages, isStreaming, currentStreamingContent } = useChatStore();
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new content (including each streamed token).
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, currentStreamingContent, isStreaming]);

  // Synthetic placeholder for the in-flight AI reply. Until the stream
  // finishes, the real AI message isn't in `messages` yet — this row carries
  // the typing indicator and the partial tokens.
  const streamingPlaceholder = isStreaming
    ? {
        id: "__streaming__",
        conversation_id: "",
        role: "ai" as const,
        content: "",
        create_time: new Date().toISOString(),
      }
    : null;

  const hasContent = messages.length > 0 || streamingPlaceholder !== null;

  return (
    <div className="flex flex-col h-full">
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        {!hasContent ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
              <span className="text-2xl">💬</span>
            </div>
            <h3 className="text-lg font-medium">Start a conversation</h3>
            <p className="text-muted-foreground mt-1 max-w-md">
              Send a message to begin chatting with your AI agent. Your agent will use the linked knowledge bases to provide informed responses.
            </p>
          </div>
        ) : (
          <div className="space-y-1">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {streamingPlaceholder && (
              <MessageBubble
                key={streamingPlaceholder.id}
                message={streamingPlaceholder}
                isStreaming
                streamingContent={currentStreamingContent}
              />
            )}
          </div>
        )}
      </ScrollArea>
      <ChatInput onSend={onSendMessage} isLoading={isLoading} />
    </div>
  );
}
