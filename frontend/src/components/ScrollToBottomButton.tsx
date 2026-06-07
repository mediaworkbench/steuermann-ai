import { Button } from "@/components/ui/button";
import { ChevronDown } from "lucide-react";

interface ScrollToBottomButtonProps {
  visible: boolean;
  unreadCount: number;
  onClick: () => void;
}

export function ScrollToBottomButton({ visible, unreadCount, onClick }: ScrollToBottomButtonProps) {
  const label =
    unreadCount > 0
      ? `${unreadCount} new message${unreadCount > 1 ? "s" : ""}. Scroll to bottom.`
      : "Scroll to bottom";

  return (
    <div
      aria-live="polite"
      aria-atomic="true"
      className={`transition-all duration-200 ${
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2 pointer-events-none"
      }`}
    >
      <Button
        onClick={onClick}
        aria-label={label}
        variant="primary"
        size="lg"
        className="relative shadow-md"
      >
        {unreadCount > 0 && (
          <span
            aria-hidden="true"
            className="absolute -top-2 -right-1 min-w-4.5 h-4.5 flex items-center justify-center rounded-full border border-border bg-surface text-primary text-[10px] font-bold px-1 leading-none"
          >
            {unreadCount > 99 ? "99+" : unreadCount}
          </span>
        )}
        <ChevronDown size={18} aria-hidden="true" />
        <span className="text-xs font-medium">Latest</span>
      </Button>
    </div>
  );
}
