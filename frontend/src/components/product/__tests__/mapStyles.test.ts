import { toMapLibreColor } from "@/components/product/mapStyles";

describe("toMapLibreColor", () => {
  const originalGetContext = HTMLCanvasElement.prototype.getContext;

  afterEach(() => {
    HTMLCanvasElement.prototype.getContext = originalGetContext;
  });

  function mockCanvas(pixel: [number, number, number, number]) {
    const ctx = {
      fillStyle: "",
      fillRect: jest.fn(),
      getImageData: jest.fn(() => ({ data: new Uint8ClampedArray(pixel) })),
    };
    HTMLCanvasElement.prototype.getContext = jest.fn(
      () => ctx,
    ) as unknown as typeof HTMLCanvasElement.prototype.getContext;
    return ctx;
  }

  it.each(["#2563eb", "rgb(37 99 235)", "rgba(37, 99, 235, 0.5)", "hsl(221 83% 53%)"])(
    "passes MapLibre-parseable %s through unchanged",
    (value) => {
      const ctx = mockCanvas([0, 0, 0, 255]);
      expect(toMapLibreColor(value, "#000000")).toBe(value);
      expect(ctx.fillRect).not.toHaveBeenCalled();
    },
  );

  it("converts lab() theme tokens to rgba via the canvas", () => {
    const ctx = mockCanvas([79, 70, 229, 255]);
    expect(toMapLibreColor("lab(39.2214% 26.9192 -70.8472)", "#6366f1")).toBe(
      "rgba(79, 70, 229, 1)",
    );
    expect(ctx.fillStyle).toBe("lab(39.2214% 26.9192 -70.8472)");
  });

  it("returns the fallback when no 2D context is available", () => {
    HTMLCanvasElement.prototype.getContext = jest.fn(
      () => null,
    ) as unknown as typeof HTMLCanvasElement.prototype.getContext;
    expect(toMapLibreColor("oklch(0.5 0.2 270)", "#6366f1")).toBe("#6366f1");
  });
});
