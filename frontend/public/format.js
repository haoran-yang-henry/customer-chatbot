// Answers are plain text with **bold** and links. They are turned into DOM nodes, never into
// HTML strings: model output must not be able to inject markup.

const PATTERN = /(\*\*[^*\n]+\*\*|https:\/\/[^\s<>"„“]+[^\s<>"„“.,;:!?)])/g;

/** [{type: "text" | "bold" | "link", value}] */
export function tokens(text) {
  const result = [];
  let last = 0;
  for (const match of text.matchAll(PATTERN)) {
    if (match.index > last) result.push({ type: "text", value: text.slice(last, match.index) });
    const value = match[0];
    result.push(value.startsWith("**") ? { type: "bold", value: value.slice(2, -2) } : { type: "link", value });
    last = match.index + value.length;
  }
  if (last < text.length) result.push({ type: "text", value: text.slice(last) });
  return result;
}

export function render(element, text) {
  element.replaceChildren(
    ...tokens(text).map(({ type, value }) => {
      if (type === "text") return document.createTextNode(value);
      const node = document.createElement(type === "bold" ? "strong" : "a");
      node.textContent = value;
      if (type === "link") {
        node.href = value;
        node.target = "_blank";
        node.rel = "noopener noreferrer";
      }
      return node;
    }),
  );
}
