"use client";

import { useState } from "react";

import { rateMemory } from "@/lib/api";

interface MemoryRatingProps {
  memoryId: string;
  initialRating?: number;
  onRatingChange?: (rating: number) => void;
}

export function MemoryRating({ memoryId, initialRating = 0, onRatingChange }: MemoryRatingProps) {
  const [rating, setRating] = useState(initialRating);
  const [saving, setSaving] = useState(false);

  async function handleRate(nextRating: number) {
    const previousRating = rating;
    setRating(nextRating);
    onRatingChange?.(nextRating);
    setSaving(true);

    const ok = await rateMemory(memoryId, nextRating);
    if (!ok) {
      setRating(previousRating);
      onRatingChange?.(previousRating);
    }

    setSaving(false);
  }

  return (
    <div className="inline-flex items-center gap-0.5" aria-label="Rate memory">
      {[1, 2, 3, 4, 5].map((star) => {
        const isFilled = star <= rating;
        return (
          <button
            key={star}
            type="button"
            onClick={() => handleRate(star)}
            disabled={saving}
            aria-label={`Rate ${star} star${star === 1 ? "" : "s"}`}
            className={[
              "cursor-pointer rounded px-0.5 text-base leading-none transition-colors",
              isFilled ? "text-amber-500" : "text-gray-300",
              "hover:text-amber-400",
              "disabled:cursor-not-allowed disabled:opacity-60",
            ].join(" ")}
          >
            ★
          </button>
        );
      })}
    </div>
  );
}
