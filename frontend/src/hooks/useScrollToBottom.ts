import { useCallback, useEffect, useRef, useState } from "react";

interface UseScrollToBottomReturn {
  scrollContainerRef: React.RefObject<HTMLDivElement | null>;
  messagesEndRef: React.RefObject<HTMLDivElement | null>;
  isAtBottom: boolean;
  unreadCount: number;
  scrollToBottom: (behavior?: ScrollBehavior) => void;
  shouldAutoScroll: boolean;
}

export function useScrollToBottom(messageCount: number): UseScrollToBottomReturn {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);
  const [unreadCount, setUnreadCount] = useState(0);
  const prevMessageCountRef = useRef(messageCount);

  // IntersectionObserver: watch the bottom sentinel against the scroll container
  useEffect(() => {
    const endEl = messagesEndRef.current;
    const containerEl = scrollContainerRef.current;
    if (!endEl || !containerEl) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsAtBottom(entry.isIntersecting);
      },
      { root: containerEl, threshold: 0.1 },
    );

    observer.observe(endEl);
    return () => observer.disconnect();
  }, []);

  // Reset unread count when user returns to bottom
  useEffect(() => {
    if (isAtBottom) {
      setUnreadCount(0);
    }
  }, [isAtBottom]);

  // Track new committed messages that arrive while scrolled up
  useEffect(() => {
    const delta = messageCount - prevMessageCountRef.current;
    if (!isAtBottom && delta > 0) {
      setUnreadCount((prev) => prev + delta);
    }
    prevMessageCountRef.current = messageCount;
  }, [messageCount, isAtBottom]);

  const scrollToBottom = useCallback((behavior: ScrollBehavior = "smooth") => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  }, []);

  return {
    scrollContainerRef,
    messagesEndRef,
    isAtBottom,
    unreadCount,
    scrollToBottom,
    shouldAutoScroll: isAtBottom,
  };
}
