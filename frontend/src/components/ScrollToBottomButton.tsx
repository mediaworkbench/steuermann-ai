import { Button } from "@/components/ui/Button";
import { Icon } from "@/components/Icon";

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
            className="absolute -top-2 -right-1 min-w-[18px] h-[18px] flex items-center justify-center rounded-full bg-white text-evergreen text-[10px] font-bold px-1 leading-none"
          >
            {unreadCount > 99 ? "99+" : unreadCount}
          </span>
        )}
        <Icon name="keyboard_arrow_down" size={18} ariaHidden />
        <span className="text-xs font-medium">Latest</span>
      </Button>
    </div>
  );
}
