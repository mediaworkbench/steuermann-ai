import { LoginScreen } from "@/components/LoginScreen";

interface LoginPageProps {
  searchParams?: Promise<{ next?: string }>;
}

function sanitizeNextPath(path?: string): string {
  if (!path || !path.startsWith("/") || path.startsWith("//")) {
    return "/";
  }
  return path;
}

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const params = searchParams ? await searchParams : undefined;
  return <LoginScreen nextPath={sanitizeNextPath(params?.next)} />;
}