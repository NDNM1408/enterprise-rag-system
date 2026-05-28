import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Message, AgenticIter } from "@/types";

// ---------------------------------------------------------------------------
//  Sidebar collapse state — persists across navigation.
// ---------------------------------------------------------------------------

interface SidebarStore {
  isCollapsed: boolean;
  toggle: () => void;
  setCollapsed: (collapsed: boolean) => void;
}

export const useSidebarStore = create<SidebarStore>()(
  persist(
    (set) => ({
      isCollapsed: false,
      toggle: () => set((s) => ({ isCollapsed: !s.isCollapsed })),
      setCollapsed: (isCollapsed) => set({ isCollapsed }),
    }),
    { name: "sidebar-store" },
  ),
);

// ---------------------------------------------------------------------------
//  Chat conversation state — current conversation + draft message.
// ---------------------------------------------------------------------------

interface ChatStore {
  conversationId: string | null;
  draft: string;
  isStreaming: boolean;
  messages: Message[];
  currentStreamingContent: string;
  currentStreamingThinking: string;
  currentStreamingAgentic: AgenticIter[];
  setConversationId: (id: string | null) => void;
  setDraft: (text: string) => void;
  setStreaming: (streaming: boolean) => void;
  addMessage: (msg: Message) => void;
  setMessages: (msgs: Message[]) => void;
  setStreamingContent: (content: string) => void;
  setStreamingThinking: (thinking: string) => void;
  appendAgenticIter: (iter: AgenticIter) => void;
  setStreamingAgentic: (iters: AgenticIter[]) => void;
  reset: () => void;
}

export const useChatStore = create<ChatStore>()((set) => ({
  conversationId: null,
  draft: "",
  isStreaming: false,
  messages: [],
  currentStreamingContent: "",
  currentStreamingThinking: "",
  currentStreamingAgentic: [],
  setConversationId: (conversationId) => set({ conversationId }),
  setDraft: (draft) => set({ draft }),
  setStreaming: (isStreaming) => set({ isStreaming }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  setMessages: (messages) => set({ messages }),
  setStreamingContent: (currentStreamingContent) => set({ currentStreamingContent }),
  setStreamingThinking: (currentStreamingThinking) => set({ currentStreamingThinking }),
  appendAgenticIter: (iter) =>
    set((s) => ({ currentStreamingAgentic: [...s.currentStreamingAgentic, iter] })),
  setStreamingAgentic: (currentStreamingAgentic) => set({ currentStreamingAgentic }),
  reset: () =>
    set({
      conversationId: null,
      draft: "",
      isStreaming: false,
      messages: [],
      currentStreamingContent: "",
      currentStreamingThinking: "",
      currentStreamingAgentic: [],
    }),
}));
