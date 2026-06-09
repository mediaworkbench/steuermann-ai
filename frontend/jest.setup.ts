import "@testing-library/jest-dom";
import { toHaveNoViolations } from "jest-axe";
import { ReadableStream, WritableStream, TransformStream } from "stream/web";

expect.extend(toHaveNoViolations);
import { TextEncoder, TextDecoder } from "util";
if (!global.ReadableStream) global.ReadableStream = ReadableStream;
if (!global.WritableStream) global.WritableStream = WritableStream;
if (!global.TransformStream) global.TransformStream = TransformStream;
if (!global.TextEncoder) global.TextEncoder = TextEncoder;
if (!global.TextDecoder) global.TextDecoder = TextDecoder;

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

global.ResizeObserver = ResizeObserverMock as unknown as typeof ResizeObserver;
