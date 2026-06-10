import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";
import jsxA11y from "eslint-plugin-jsx-a11y";

export default defineConfig([
  ...nextVitals,
  ...nextTs,
  // jsx-a11y plugin is already registered by eslint-config-next/core-web-vitals (6 rules);
  // add the remaining recommended rules without re-registering the plugin.
  { rules: jsxA11y.flatConfigs.recommended.rules },
  {
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "react-hooks/set-state-in-effect": "off",
    },
  },
  /* ─────────────────────────────────────────────────────────────────────────
   * Design-system enforcement gates
   * These rules enforce the locked choices from design-system.md:
   *   1. Icon system: direct lucide-react imports are the canonical path.
   *      The legacy `material-symbols-outlined` CSS class is blocked.
   *   2. Token policy: raw Tailwind palette utilities (any color family with a
   *      numeric shade, e.g. bg-blue-500, text-yellow-500) are blocked across
   *      all of src/. Use the semantic token classes instead:
   *        emerald, green  →  success
   *        amber, yellow   →  warning
   *        red             →  destructive
   *        blue            →  info
   *      Semantic chart tokens (text-chart-amber etc.) are unaffected.
   * ───────────────────────────────────────────────────────────────────────── */
  {
    files: ["src/**/*.{ts,tsx}"],
    rules: {
      // Gate 1 – icon system: block the legacy material-symbols-outlined class
      // Gate 2 – token policy: block raw palette status-color class strings
      "no-restricted-syntax": [
        "error",
        {
          selector:
            "Literal[value=/material-symbols-outlined/]",
          message:
            "Material Symbols is no longer part of the design system. " +
            "Use lucide-react directly instead.",
        },
        {
          selector:
            "Literal[value=/\\b(bg|text|border|ring|stroke|fill|from|to|via)-(slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\\d{1,3}(\\/\\d{1,3})?\\b/]",
          message:
            "Raw Tailwind palette utilities are not allowed. " +
            "Use semantic tokens instead: success / warning / destructive / info / surface / muted.",
        },
        {
          selector:
            "JSXAttribute[name.name='className'] Literal[value=/#[0-9a-fA-F]{3,8}/]",
          message:
            "Hex color literals in JSX className are not allowed. Use semantic token classes or CSS variables.",
        },
      ],
    },
  },
  {
    files: ["src/components/ui/**/*.{ts,tsx}"],
    rules: {
      // Shared UI primitives are foundational and must remain page-agnostic.
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@/app/*", "./app/*", "../app/*", "../../app/*", "../../../app/*", "../../../../app/*"],
              message:
                "UI primitives must not import from app-layer modules. Move shared logic into components/ui, components/product, or lib.",
            },
            {
              group: [
                "@/components/product/*",
                "./product/*",
                "../product/*",
                "../../product/*",
                "../../../product/*",
              ],
              message:
                "UI primitives must not depend on product-layer components. Keep ui as the lowest-level shared component layer.",
            },
          ],
        },
      ],
    },
  },
  {
    files: ["src/components/product/**/*.{ts,tsx}"],
    rules: {
      // Product components are shared across routes and must not depend on page-layer code.
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["@/app/*", "./app/*", "../app/*", "../../app/*", "../../../app/*", "../../../../app/*"],
              message:
                "Product components must not import from app-layer modules. Lift shared logic to product/ui/lib and keep pages as composition only.",
            },
          ],
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
