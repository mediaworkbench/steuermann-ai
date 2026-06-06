import * as React from "react";
import { cn } from "@/lib/utils";
import { Icon } from "@/components/Icon";

type ChatRole = "assistant" | "user";

interface ChatMessageShellProps {
  role: ChatRole;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}

export function ChatMessageShell({ role, children, className, bodyClassName }: ChatMessageShellProps) {
  const assistant = role === "assistant";

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
        <Icon
          name={assistant ? "smart_toy" : "person"}
          size={18}
          className={assistant ? "text-background" : "text-primary-foreground"}
        />
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
