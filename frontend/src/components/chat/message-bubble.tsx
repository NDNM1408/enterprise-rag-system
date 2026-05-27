"use client";

import { cn } from "@/lib/utils";
import type { Message } from "@/types";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Bot, User, Copy, Check, Brain, ChevronDown, ChevronRight } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
  streamingContent?: string;
  streamingThinking?: string;
}

interface ThinkingBlockProps {
  thinking: string;
  isStreaming: boolean;
  hasContent: boolean;
}

function ThinkingBlock({ thinking, isStreaming, hasContent }: ThinkingBlockProps) {
  // Open while thinking is in-flight (no content yet); collapse once the
  // model starts emitting the answer. User can toggle anytime.
  const [open, setOpen] = useState(true);
  useEffect(() => {
    if (hasContent && isStreaming) {
      setOpen(false);
    }
  }, [hasContent, isStreaming]);

  const label = isStreaming && !hasContent ? "Thinking…" : "Thoughts";

  return (
    <div className="rounded-lg border border-border bg-muted/30 text-xs">
      <button
        type="button"
        onClick={() => setOpen((s) => !s)}
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? (
          <ChevronDown className="h-3 w-3" />
        ) : (
          <ChevronRight className="h-3 w-3" />
        )}
        <Brain className="h-3 w-3" />
        <span className="font-medium">{label}</span>
        {isStreaming && !hasContent && (
          <span className="ml-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
        )}
      </button>
      {open && (
        <div className="border-t border-border/50 px-3 py-2 text-muted-foreground whitespace-pre-wrap leading-relaxed max-h-64 overflow-y-auto">
          {thinking}
        </div>
      )}
    </div>
  );
}

// Approximates the `prose` plugin: spacing + table + code formatting so AI
// responses look like proper markdown without pulling in the typography
// package. Shared shape mirrors document-preview-dialog for consistency.
const markdownClasses = cn(
  "max-w-none text-sm leading-relaxed",
  "[&>:first-child]:mt-0 [&>:last-child]:mb-0",
  "[&_h1]:text-lg [&_h1]:font-bold [&_h1]:mt-4 [&_h1]:mb-2",
  "[&_h2]:text-base [&_h2]:font-semibold [&_h2]:mt-3 [&_h2]:mb-2",
  "[&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mt-3 [&_h3]:mb-1",
  "[&_p]:my-2",
  "[&_ul]:list-disc [&_ul]:pl-5 [&_ul]:my-2",
  "[&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:my-2",
  "[&_li]:my-0.5",
  "[&_a]:text-primary [&_a]:underline",
  "[&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-xs [&_code]:font-mono",
  "[&_pre]:rounded [&_pre]:bg-muted [&_pre]:p-3 [&_pre]:my-3 [&_pre]:overflow-x-auto",
  "[&_pre_code]:bg-transparent [&_pre_code]:p-0",
  "[&_blockquote]:border-l-4 [&_blockquote]:border-muted [&_blockquote]:pl-3 [&_blockquote]:italic [&_blockquote]:text-muted-foreground",
  "[&_hr]:my-3 [&_hr]:border-border",
  "[&_table]:my-3 [&_table]:w-full [&_table]:border-collapse [&_table]:text-xs",
  "[&_th]:border [&_th]:border-border [&_th]:bg-muted/50 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_th]:font-medium",
  "[&_td]:border [&_td]:border-border [&_td]:px-2 [&_td]:py-1 [&_td]:align-top",
  "[&_strong]:font-semibold",
);

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1 py-1">
      <span className="h-2 w-2 rounded-full bg-current opacity-60 animate-bounce [animation-delay:-0.3s]" />
      <span className="h-2 w-2 rounded-full bg-current opacity-60 animate-bounce [animation-delay:-0.15s]" />
      <span className="h-2 w-2 rounded-full bg-current opacity-60 animate-bounce" />
    </span>
  );
}

export function MessageBubble({
  message,
  isStreaming,
  streamingContent,
  streamingThinking,
}: MessageBubbleProps) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "human";
  // While streaming, show the live buffer; the persisted message.content is
  // empty until the stream finishes.
  const content = isStreaming ? streamingContent || "" : message.content;
  const thinking = isStreaming ? streamingThinking || "" : message.thinking || "";
  // Show typing dots only when neither answer nor thinking has arrived yet.
  const showTypingDots = isStreaming && !content && !thinking;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={cn(
        "flex gap-3 py-4",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      <Avatar className="h-8 w-8">
        <AvatarFallback
          className={cn(
            isUser ? "bg-primary text-primary-foreground" : "bg-muted"
          )}
        >
          {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
        </AvatarFallback>
      </Avatar>
      <div
        className={cn(
          "flex flex-col gap-2 max-w-[80%]",
          isUser ? "items-end" : "items-start"
        )}
      >
        {!isUser && thinking && (
          <ThinkingBlock
            thinking={thinking}
            isStreaming={!!isStreaming}
            hasContent={!!content}
          />
        )}
        {(showTypingDots || content || isUser) && (
          <div
            className={cn(
              "rounded-2xl px-4 py-2.5",
              isUser
                ? "bg-primary text-primary-foreground"
                : "bg-muted"
            )}
          >
            {showTypingDots ? (
              <TypingDots />
            ) : isUser ? (
              <p className="text-sm whitespace-pre-wrap">{content}</p>
            ) : (
              <div className={markdownClasses}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {content}
                </ReactMarkdown>
                {isStreaming && (
                  <span className="inline-block w-2 h-4 bg-current animate-pulse ml-1 align-middle" />
                )}
              </div>
            )}
          </div>
        )}
        {!isUser && !isStreaming && content && (
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-muted-foreground"
            onClick={handleCopy}
          >
            {copied ? (
              <>
                <Check className="h-3 w-3 mr-1" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-3 w-3 mr-1" />
                Copy
              </>
            )}
          </Button>
        )}
      </div>
    </div>
  );
}
