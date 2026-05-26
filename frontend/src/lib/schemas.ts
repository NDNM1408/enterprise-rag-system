import { z } from "zod";

// ---------------------------------------------------------------------------
//  Knowledge base creation
// ---------------------------------------------------------------------------

export const createKbSchema = z.object({
  name: z.string().min(1, "Name is required").max(120),
  description: z.string().max(500).optional(),
  parser_config: z
    .object({
      rag_mode: z.enum(["classic", "llm-wiki"]).optional(),
    })
    .optional(),
});

export type CreateKbInput = z.infer<typeof createKbSchema>;

// ---------------------------------------------------------------------------
//  Agent creation
// ---------------------------------------------------------------------------

export const createAgentSchema = z.object({
  name: z.string().min(1, "Name is required").max(120),
  description: z.string().max(500).optional(),
  llm_model: z.string().min(1, "Model is required"),
  llm_temperature: z.number().min(0).max(2).optional(),
  system_prompt: z.string().max(8000).optional(),
});

export type CreateAgentInput = z.infer<typeof createAgentSchema>;

// ---------------------------------------------------------------------------
//  Login
// ---------------------------------------------------------------------------

export const loginSchema = z.object({
  email: z.string().email("Invalid email"),
  password: z.string().min(6, "Password must be at least 6 characters"),
});

export type LoginInput = z.infer<typeof loginSchema>;
