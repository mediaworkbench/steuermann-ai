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
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
} from "@/components/ui/sidebar";
import { Slider } from "@/components/ui/slider";
import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider } from "@/hooks/useTheme";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

beforeAll(() => {
  // jsdom lacks matchMedia (needed by sidebar's useIsMobile and sonner) and
  // scrollIntoView (used by base-ui menu focus management).
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: jest.fn(),
      removeListener: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }),
  });
  Element.prototype.scrollIntoView = jest.fn();
});

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

  // Overlay primitives render into portals, so axe runs on baseElement.
  // Two rules are disabled for these tests only:
  // - aria-command-name: base-ui's internal focus-guard spans (role="button",
  //   data-base-ui-focus-guard) have no accessible name by design; native
  //   buttons remain covered by the separate button-name rule.
  // - region: a component test is not a full page, so portal content is
  //   never inside a landmark.
  const overlayAxeOptions = {
    rules: {
      "aria-command-name": { enabled: false },
      region: { enabled: false },
    },
  };
  test("AlertDialog (open) has no violations", async () => {
    const { baseElement } = render(
      <AlertDialog open>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete item</AlertDialogTitle>
            <AlertDialogDescription>This action cannot be undone.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    );
    expect(await axe(baseElement, overlayAxeOptions)).toHaveNoViolations();
  });

  test("Avatar with fallback has no violations", async () => {
    const { container } = render(
      <Avatar>
        <AvatarFallback>AB</AvatarFallback>
      </Avatar>
    );
    expect(await axe(container)).toHaveNoViolations();
  });

  test("DropdownMenu (open) has no violations", async () => {
    const { baseElement } = render(
      <DropdownMenu open>
        <DropdownMenuTrigger>Options</DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuGroup>
            <DropdownMenuLabel>Actions</DropdownMenuLabel>
            <DropdownMenuItem>Edit</DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem>Delete</DropdownMenuItem>
          </DropdownMenuGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    );
    expect(await axe(baseElement, overlayAxeOptions)).toHaveNoViolations();
  });

  test("Popover (open) has no violations", async () => {
    const { baseElement } = render(
      <Popover open>
        <PopoverTrigger>Show details</PopoverTrigger>
        <PopoverContent>
          <PopoverHeader>
            <PopoverTitle>Details</PopoverTitle>
            <PopoverDescription>More information.</PopoverDescription>
          </PopoverHeader>
        </PopoverContent>
      </Popover>
    );
    expect(await axe(baseElement, overlayAxeOptions)).toHaveNoViolations();
  });

  test("Sheet (open) has no violations", async () => {
    const { baseElement } = render(
      <Sheet open>
        <SheetContent>
          <SheetHeader>
            <SheetTitle>Panel</SheetTitle>
            <SheetDescription>Side panel content.</SheetDescription>
          </SheetHeader>
        </SheetContent>
      </Sheet>
    );
    expect(await axe(baseElement, overlayAxeOptions)).toHaveNoViolations();
  });

  test("Sidebar has no violations", async () => {
    const { container } = render(
      <SidebarProvider>
        <Sidebar>
          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupLabel>Navigation</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton>Home</SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>
        </Sidebar>
      </SidebarProvider>
    );
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Slider with aria-label has no violations", async () => {
    const { container } = render(<Slider aria-label="Volume" defaultValue={50} />);
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Toaster has no violations", async () => {
    const { baseElement } = render(<ThemeProvider><Toaster /></ThemeProvider>);
    expect(await axe(baseElement, overlayAxeOptions)).toHaveNoViolations();
  });

  test("Table has no violations", async () => {
    const { container } = render(
      <Table>
        <TableCaption>Monthly usage</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Value</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell>Tokens</TableCell>
            <TableCell>1200</TableCell>
          </TableRow>
        </TableBody>
      </Table>
    );
    expect(await axe(container)).toHaveNoViolations();
  });

  test("Tooltip (open) has no violations", async () => {
    const { baseElement } = render(
      <TooltipProvider>
        <Tooltip open>
          <TooltipTrigger>Info</TooltipTrigger>
          <TooltipContent>Helpful hint</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
    expect(await axe(baseElement, overlayAxeOptions)).toHaveNoViolations();
  });
});
