# Frontend Design System Directive

This document is the implementation directive for all future frontend work in this repository.

It defines the non-negotiable usage rules for design tokens, shared components, styling boundaries, and accessibility behavior.

## Scope

This directive applies to all frontend code under `frontend/src`, including:

- route files in `app/`
- shared UI primitives in `components/ui/`
- shared product components in `components/product/`
- feature components in `components/`
- styles in `app/globals.css` and any approved module-local styles

## Authoritative Layers

Use these layers exactly as defined below.

1. Foundation layer
   - semantic tokens and global style contract
   - source: `frontend/src/app/globals.css`

2. UI primitive layer
   - reusable base primitives (button/input/dialog/etc.)
   - source: `frontend/src/components/ui/`

3. Product component layer
   - reusable app-level composition blocks
   - source: `frontend/src/components/product/`

4. Page composition layer
   - route assembly only
   - source: `frontend/src/app/**/page.tsx`

Page code must compose shared components. It must not recreate reusable primitives.

## Mandatory Rules

### 1) Primitive system requirement

- All new or modified reusable UI elements must follow the Radix primitives + shadcn/ui system used in this repository.
- Build on existing primitives in `components/ui/*` before creating new abstractions.
- Do not introduce parallel primitive systems or ad hoc replacements for shared primitives.

### 2) Token policy

- Use semantic token classes and CSS variables.
- Do not introduce raw palette utility classes in shared components.
- Do not introduce hex color literals in JSX `className` strings.

Use semantic roles such as:

- `success`
- `warning`
- `destructive`
- `info`
- `surface`, `surface-muted`, `border`, `foreground`, `muted-foreground`

### 3) Shared-component boundaries

- `components/ui/*` is the lowest shared layer.
- UI primitives must not import from `app/*`.
- UI primitives must not import from `components/product/*`.
- Product components must not import from `app/*`.

When shared logic is needed, move it to:

- `components/ui/*`
- `components/product/*`
- `lib/*`

### 4) Icon system

- Canonical icon path: `components/Icon.tsx` (Material Symbols).
- Direct `lucide-react` imports are not allowed.

### 5) Accessibility baseline

Every new or changed interactive element must support:

- full keyboard interaction
- visible focus state
- semantic roles and labels
- predictable escape and dismiss behavior for overlays

## ESLint Enforcement

The directive is enforced in `frontend/eslint.config.mjs`.

Current enforced checks include:

- blocked `lucide-react` imports
- blocked raw status/palette class patterns in JSX class strings
- blocked hex color literals in JSX class strings
- import-boundary restrictions between `ui`, `product`, and `app` layers

These checks are treated as quality gates and must pass in local lint runs and CI-equivalent enforcement.

## Implementation Workflow For New UI

Use this workflow for every UI change.

1. Start from existing shared primitives/components.
2. If a pattern appears twice, extract it into a shared component.
3. Use semantic tokens only for visual styling.
4. Keep route files focused on composition and data flow.
5. Validate with `npm run lint` and relevant tests.

## Do / Do Not

### Do

- Prefer composition with `components/ui` and `components/product`.
- Reuse existing semantic utility classes and token variables.
- Keep behavior and visuals profile-safe through token-driven styling.

### Do Not

- Add ad hoc primitive clones in page or feature folders.
- Re-introduce raw palette class usage in shared components.
- Add new icon systems or direct third-party icon imports.

## PR Review Checklist

Every frontend PR must satisfy all items below:

1. No layer-boundary violations (`ui`/`product`/`app`).
2. No raw palette or hex drift in JSX class strings.
3. No direct `lucide-react` import.
4. Reusable UI extracted to shared layers when repeated.
5. Keyboard and focus behavior validated for interactive changes.
6. `npm run lint` passes.

## Related Docs

- `design-system.md` (plan and migration history)
- `docs/profile_creation.md` (profile overlay workflow)
- `docs/configuration.md` (runtime configuration contract)
- `docs/status.md` (runtime/documentation sync checkpoint
