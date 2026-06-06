import * as React from "react";
import { cn } from "@/lib/utils";

interface PageErrorAlertProps {
  title: React.ReactNode;
  message?: React.ReactNode;
  className?: string;
}

export function PageErrorAlert({ title, message, className }: PageErrorAlertProps) {
  return (
    <div
      role="alert"
      className={cn(
        "rounded-lg border border-destructive/35 bg-destructive/10 px-4 py-3 text-destructive",
        className,
      )}
    >
      <p className="mb-1 font-semibold">{title}</p>
      {message ? <p>{message}</p> : null}
    </div>
  );
}
