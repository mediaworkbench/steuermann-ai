"use client";

import { useEffect, useState } from "react";
import { Star } from "lucide-react";

import { rateMemory } from "@/lib/api";

interface MemoryRatingProps {
  memoryId: string;
  initialRating?: number;
  onRatingChange?: (rating: number) => void;
  compact?: boolean;
  showStatus?: boolean;
  ariaLabel?: string;
  getRateLabel?: (star: number) => string;
  statusLabels?: {
    saving: string;
    saved: string;
    retry: string;
  };
}

export function MemoryRating({
  memoryId,
  initialRating = 0,
  onRatingChange,
  compact = false,
  showStatus = false,
  ariaLabel = "Rate memory usefulness for future answers",
  getRateLabel,
  statusLabels = {
    saving: "Saving",
    saved: "Saved",
    retry: "Retry",
  },
}: MemoryRatingProps) {
  const [rating, setRating] = useState(initialRating);
  const [hover, setHover] = useState(0);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<"idle" | "saved" | "error">("idle");

  useEffect(() => {
    setRating(initialRating);
  }, [initialRating]);

  async function handleRate(nextRating: number) {
    const previousRating = rating;
    setRating(nextRating);
    onRatingChange?.(nextRating);
    setSaving(true);
    setStatus("idle");

    const ok = await rateMemory(memoryId, nextRating);
    if (!ok) {
      setRating(previousRating);
      onRatingChange?.(previousRating);
      setStatus("error");
    } else {
      setStatus("saved");
    }

    setSaving(false);
  }

  return (
    <div className="inline-flex items-center gap-2" aria-label={ariaLabel}>
      {[1, 2, 3, 4, 5].map((star) => {
        const isFilled = star <= (hover || rating);
        const label = getRateLabel
          ? getRateLabel(star)
          : `Rate ${star} star${star === 1 ? "" : "s"}`;
        return (
          <button
            key={star}
            type="button"
            onClick={() => handleRate(star)}
            onMouseEnter={() => setHover(star)}
            onMouseLeave={() => setHover(0)}
            disabled={saving}
            aria-label={label}
            title={label}
            className={[
              "cursor-pointer rounded p-0.5 transition-colors",
              isFilled ? "text-yellow-500" : "text-evergreen/35",
              "hover:text-yellow-500",
              "disabled:cursor-not-allowed disabled:opacity-60",
            ].join(" ")}
          >
            <Star size={compact ? 13 : 15} className={isFilled ? "fill-yellow-400" : ""} />
          </button>
        );
      })}
      {showStatus && (
        <span aria-live="polite" className="min-w-12 text-[11px] text-evergreen/55">
          {saving
            ? statusLabels.saving
            : status === "saved"
              ? statusLabels.saved
              : status === "error"
                ? statusLabels.retry
                : ""}
        </span>
      )}
    </div>
  );
}
