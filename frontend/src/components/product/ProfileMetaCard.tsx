import * as React from "react";

interface ProfileMetaCardProps {
  heading: React.ReactNode;
  detail: React.ReactNode;
}

export function ProfileMetaCard({ heading, detail }: ProfileMetaCardProps) {
  return (
    <section>
      <h2 className="m-0 text-lg font-bold tracking-tight text-foreground">{heading}</h2>
      <p className="mt-1.5 text-sm text-muted-foreground">{detail}</p>
    </section>
  );
}
