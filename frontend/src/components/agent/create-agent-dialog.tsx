"use client";

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { createAgentSchema, type CreateAgentInput } from "@/lib/schemas";
import { useCreateAgent, useKnowledgeBases, useLiteLLMModels } from "@/lib/hooks";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2 } from "lucide-react";

interface CreateAgentDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CreateAgentDialog({ open, onOpenChange }: CreateAgentDialogProps) {
  const createAgent = useCreateAgent();
  const { data: liteLLMModels, isLoading: modelsLoading } = useLiteLLMModels();

  const form = useForm<CreateAgentInput>({
    resolver: zodResolver(createAgentSchema),
    defaultValues: {
      name: "",
      description: "",
      llm_model: "",
      llm_temperature: 0.7,
      system_prompt: "",
    },
  });

  async function onSubmit(data: CreateAgentInput) {
    try {
      await createAgent.mutateAsync(data);
      form.reset();
      onOpenChange(false);
    } catch {
      // Error is handled by the mutation
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Create Agent</DialogTitle>
          <DialogDescription>
            Create a new AI agent that can answer questions using your knowledge bases.
          </DialogDescription>
        </DialogHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input placeholder="My Agent" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="Describe what this agent does..."
                      className="resize-none"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="llm_model"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>LLM Model</FormLabel>
                  <Select onValueChange={field.onChange} defaultValue={field.value}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue
                          placeholder={
                            modelsLoading ? "Loading models…" : "Select a model"
                          }
                        />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {(liteLLMModels ?? []).map((id) => (
                        <SelectItem key={id} value={id}>
                          {id}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="llm_temperature"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Temperature: {field.value}</FormLabel>
                  <FormControl>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.1"
                      value={field.value}
                      onChange={(e) => field.onChange(Number(e.target.value))}
                      className="w-full"
                    />
                  </FormControl>
                  <FormDescription>
                    Lower values are more focused, higher values are more creative
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="system_prompt"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>System Prompt</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="You are a helpful assistant..."
                      className="resize-none min-h-[100px]"
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>
                    Optional instructions for the agent&apos;s behavior
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={createAgent.isPending}>
                {createAgent.isPending && (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                )}
                Create
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
