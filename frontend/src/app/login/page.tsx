import { LoginScreen } from "@/components/LoginScreen";
import { sanitizeNextPath } from "@/lib/auth/safeRedirect";

interface LoginPageProps {
  searchParams?: Promise<{ next?: string }>;
}

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const params = searchParams ? await searchParams : undefined;
  return <LoginScreen nextPath={sanitizeNextPath(params?.next)} />;
}