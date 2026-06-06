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
