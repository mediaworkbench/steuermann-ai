import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

export default defineConfig([
  ...nextVitals,
  ...nextTs,
  {
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "react-hooks/set-state-in-effect": "off",
    },
  },
  /* ─────────────────────────────────────────────────────────────────────────
   * Design-system enforcement gates
   * These rules enforce the locked choices from design-system.md:
   *   1. Icon system: only the shared Icon wrapper (Material Symbols) is allowed.
   *      Direct lucide-react imports are blocked everywhere in src/.
   *   2. Token policy: raw Tailwind palette status-color utilities are blocked
   *      in shared components and pages. Use the semantic token classes instead:
   *        emerald-*  →  success
   *        amber-*    →  warning
   *        red-*      →  destructive  (bg-red-*, text-red-*)
   *        green-*    →  success
   *        blue-100   →  info/10
   * ───────────────────────────────────────────────────────────────────────── */
  {
    files: ["src/**/*.{ts,tsx}"],
    rules: {
      // Gate 1 – icon system: block lucide-react re-introduction
      "no-restricted-imports": [
        "error",
        {
          name: "lucide-react",
          message:
            "lucide-react is not part of the design system. Use the shared Icon wrapper " +
            "(components/Icon.tsx) with a Material Symbols icon name instead.",
        },
      ],
      // Gate 2 – token policy: block raw palette status-color class strings
      // This catches the most common drift patterns at the JSX className level.
      "no-restricted-syntax": [
        "warn",
        {
          selector:
            "Literal[value=/\\b(bg|text|border)-(emerald|amber|red|green)-(\\d+|\\w+)\\b/]",
          message:
            "Raw palette status-color utilities are not allowed. " +
            "Use semantic tokens instead: success / warning / destructive / info.",
        },
        {
          selector: "Literal[value=/\\bbg-blue-100\\b/]",
          message:
            "Raw palette utility bg-blue-100 is not allowed. Use bg-info/10 instead.",
        },
      ],
    },
  },
  {
    files: ["jest.config.js"],
    rules: {
      "@typescript-eslint/no-require-imports": "off",
    },
  },
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
]);
