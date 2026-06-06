import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge Tailwind classes without conflicts. Use instead of raw template strings. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
