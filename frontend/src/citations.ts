/** Turn "[S1]" / "[S1, S2]" into markdown links "[S1](#cite-S1)" so they render as chips. */
export function linkCitations(text: string): string {
  return text.replace(/\[(S\d+(?:\s*,\s*S\d+)*)\]/g, (_, group: string) =>
    group
      .split(/\s*,\s*/)
      .map((id) => `[${id}](#cite-${id})`)
      .join(" "),
  );
}

export function citationId(href: string | undefined): string | null {
  const m = href?.match(/^#cite-(S\d+)$/);
  return m ? m[1] : null;
}
