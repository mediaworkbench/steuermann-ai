import * as React from "react";
import { cn } from "@/lib/utils";
import { Bot, User } from "lucide-react";

type ChatRole = "assistant" | "user";

interface ChatMessageShellProps {
  messageRole: ChatRole;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}

export function ChatMessageShell({ messageRole, children, className, bodyClassName }: ChatMessageShellProps) {
  const assistant = messageRole === "assistant";

  return (
    <div
      className={cn(
        "msg-row mx-auto flex max-w-5xl gap-4",
        !assistant && "flex-row-reverse",
        className
      )}
    >
      <div
        className={cn(
          "shrink-0",
          assistant
            ? "mt-1 flex h-8 w-8 items-center justify-center rounded-full bg-foreground"
            : "flex h-8 w-8 items-center justify-center rounded-full bg-linear-to-tr from-primary to-primary/30"
        )}
        aria-hidden="true"
      >
        {assistant ? (
          <Bot size={18} className="text-background" />
        ) : (
          <User size={18} className="text-primary-foreground" />
        )}
      </div>
      <div
        className={cn(
          "flex flex-col gap-1",
          assistant ? "w-full max-w-[85%] items-start" : "max-w-[85%] items-end",
          bodyClassName
        )}
      >
        {children}
      </div>
    </div>
  );
}
