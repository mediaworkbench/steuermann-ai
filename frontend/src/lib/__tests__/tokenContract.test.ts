import { findUnknownTokens } from "@/lib/tokenContract";

describe("findUnknownTokens", () => {
  it("returns empty for valid color tokens", () => {
    expect(findUnknownTokens("colors", ["primary", "background", "foreground"])).toEqual([]);
  });

  it("flags unknown color tokens", () => {
    expect(findUnknownTokens("colors", ["primary", "nope", "wat"])).toEqual(["nope", "wat"]);
  });

  it("returns empty for valid font tokens", () => {
    expect(findUnknownTokens("fonts", ["font-sans"])).toEqual([]);
  });

  it("flags unknown font tokens", () => {
    expect(findUnknownTokens("fonts", ["font-sans", "font-bogus"])).toEqual(["font-bogus"]);
  });

  it("returns empty for valid radius tokens", () => {
    expect(findUnknownTokens("radius", ["radius"])).toEqual([]);
  });

  it("flags unknown radius tokens", () => {
    expect(findUnknownTokens("radius", ["radius", "radius-unknown"])).toEqual(["radius-unknown"]);
  });

  it("handles empty arrays", () => {
    expect(findUnknownTokens("colors", [])).toEqual([]);
  });
});
