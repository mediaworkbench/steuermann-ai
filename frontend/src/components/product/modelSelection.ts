export function formatModelName(model: string | null | undefined, fallback = "Model"): string {
  const m = model || fallback;
  const parts = m.split("/");
  return parts.length > 1 ? parts.slice(1).join("/") : m;
}

export function updatePreferredModelSelection(
  prev: Record<string, string | null>,
  roleName: string,
  value: string,
  roleDefaultModel: string
): Record<string, string | null> {
  return {
    ...prev,
    [roleName]: value === roleDefaultModel ? "" : value,
  };
}
