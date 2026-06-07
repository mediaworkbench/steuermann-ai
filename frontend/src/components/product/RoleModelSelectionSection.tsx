import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { RoleModelSelectorCard } from "@/components/product/RoleModelSelectorCard";
import { SectionStateText } from "@/components/product/SectionStateText";

interface RoleModelConfig {
  role: string;
  provider_id?: string;
  default_model?: string;
  available_models?: string[];
  model_load_error?: string | null;
}

interface RoleModelSelectionSectionProps {
  title: string;
  loading: boolean;
  loadingLabel: string;
  emptyLabel: string;
  roleConfigs: RoleModelConfig[];
  preferredModels: Record<string, string | null>;
  onModelChange: (roleName: string, value: string, roleDefaultModel: string) => void;
  t: (key: string, params?: Record<string, string>) => string;
}

export function RoleModelSelectionSection({
  title,
  loading,
  loadingLabel,
  emptyLabel,
  roleConfigs,
  preferredModels,
  onModelChange,
  t,
}: RoleModelSelectionSectionProps) {
  return (
    <Card>
      <CardHeader className="px-0 pt-0">
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      {loading ? (
        <SectionStateText>{loadingLabel}</SectionStateText>
      ) : roleConfigs.length === 0 ? (
        <SectionStateText>{emptyLabel}</SectionStateText>
      ) : (
        <div className="space-y-4">
          {roleConfigs.map((roleConfig) => {
            const roleName = roleConfig.role;
            const roleDefaultModel = roleConfig.default_model || "";
            const selectedModel = preferredModels[roleName] || roleDefaultModel;
            const roleModels = roleConfig.available_models || [];
            const mergedRoleModels = roleModels.includes(roleDefaultModel)
              ? roleModels
              : [roleDefaultModel, ...roleModels].filter(Boolean);

            return (
              <RoleModelSelectorCard
                key={roleName}
                roleName={roleName}
                roleLabel={t("settingsPanel.roleModelLabel", { role: roleName })}
                providerLabel={
                  t("settingsPanel.roleProviderLocked", { provider: roleConfig.provider_id || "" })
                }
                systemDefaultLabel={t("settingsPanel.systemDefault", { value: roleDefaultModel })}
                selectedModel={selectedModel}
                modelOptions={mergedRoleModels}
                modelLoadError={roleConfig.model_load_error}
                onModelChange={(value) => onModelChange(roleName, value, roleDefaultModel)}
              />
            );
          })}
        </div>
      )}
    </Card>
  );
}
