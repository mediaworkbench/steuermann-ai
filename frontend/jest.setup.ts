import "@testing-library/jest-dom";

// Polyfill Web Streams API for jsdom (Node 18+ has these in stream/web and util)
const { ReadableStream, WritableStream, TransformStream } = require("stream/web");
const { TextEncoder, TextDecoder } = require("util");
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
