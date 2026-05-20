import { create } from "zustand";
import { persist } from "zustand/middleware";

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
  setConversationId: (id: string | null) => void;
  setDraft: (text: string) => void;
  reset: () => void;
}

export const useChatStore = create<ChatStore>()((set) => ({
  conversationId: null,
  draft: "",
  setConversationId: (conversationId) => set({ conversationId }),
  setDraft: (draft) => set({ draft }),
  reset: () => set({ conversationId: null, draft: "" }),
}));
