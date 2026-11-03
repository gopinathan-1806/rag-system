import { citationId, linkCitations } from "./citations";
import { parseSse } from "./api";

describe("citations", () => {
  it("links single and grouped citations", () => {
    expect(linkCitations("Python [S1]. Go [S2, S3].")).toBe(
      "Python [S1](#cite-S1). Go [S2](#cite-S2) [S3](#cite-S3).",
    );
  });
  it("only accepts internal citation hrefs", () => {
    expect(citationId("#cite-S4")).toBe("S4");
    expect(citationId("https://evil.example")).toBeNull();
    expect(citationId(undefined)).toBeNull();
  });
});

describe("parseSse", () => {
  it("reassembles events split across chunks", async () => {
    const enc = new TextEncoder();
    const parts = ['data: {"type":"tok', 'en","text":"hi"}\n\ndata: {"type":"done","cited":[],"grounded":false}\n\n'];
    const stream = new ReadableStream<Uint8Array>({
      start(c) { parts.forEach((p) => c.enqueue(enc.encode(p))); c.close(); },
    });
    const out = [];
    for await (const e of parseSse(stream)) out.push(e);
    expect(out.map((e) => e.type)).toEqual(["token", "done"]);
  });
});
