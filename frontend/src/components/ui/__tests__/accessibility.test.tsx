import { render } from "@testing-library/react";
import { axe } from "jest-axe";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardTitle, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

describe("ui/ component accessibility", () => {
  test("Button has no violations", async () => {
    const { container } = render(<Button>Click me</Button>);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Input with aria-label has no violations", async () => {
    const { container } = render(<Input aria-label="Name" />);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Textarea with aria-label has no violations", async () => {
    const { container } = render(<Textarea aria-label="Message" />);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Select with aria-label has no violations", async () => {
    const { container } = render(
      <Select aria-label="Pick one">
        <option value="a">A</option>
        <option value="b">B</option>
      </Select>
    );
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Checkbox with aria-label has no violations", async () => {
    const { container } = render(<Checkbox aria-label="Accept terms" />);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Switch with aria-label has no violations", async () => {
    const { container } = render(<Switch aria-label="Toggle setting" />);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Label with htmlFor has no violations", async () => {
    const { container } = render(<Label htmlFor="x">Name</Label>);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Badge has no violations", async () => {
    const { container } = render(<Badge>New</Badge>);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Alert with title and description has no violations", async () => {
    const { container } = render(
      <Alert>
        <AlertTitle>Heads up</AlertTitle>
        <AlertDescription>Something happened.</AlertDescription>
      </Alert>
    );
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Separator has no violations", async () => {
    const { container } = render(<Separator />);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Skeleton has no violations", async () => {
    const { container } = render(<Skeleton className="h-10 w-10" />);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Card with title has no violations", async () => {
    const { container } = render(
      <Card>
        <CardTitle>Section</CardTitle>
        <CardContent>Content</CardContent>
      </Card>
    );
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Tabs has no violations", async () => {
    const { container } = render(
      <Tabs defaultValue="a">
        <TabsList aria-label="Sections">
          <TabsTrigger value="a">Tab A</TabsTrigger>
          <TabsTrigger value="b">Tab B</TabsTrigger>
        </TabsList>
        <TabsContent value="a">Content A</TabsContent>
        <TabsContent value="b">Content B</TabsContent>
      </Tabs>
    );
    expect(await axe(container)).toHaveNoViolations();
  });
});
