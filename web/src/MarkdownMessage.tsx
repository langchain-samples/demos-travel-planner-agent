import type { Components } from "react-markdown";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Props = {
  children: string;
  isDark: boolean;
};

const BULLET = /\u2022/; // •

/**
 * Unicode bullets (•) are not Markdown list markers. Normalize so lists render
 * as block `<ul>` (handles legacy one-line prompts like "with: • a • b").
 */
function normalizeUnicodeBulletsToMarkdown(src: string): string {
  let s = src;
  // "Label: • item" → break before first list item
  s = s.replace(/:\s*\u2022\s+/g, ":\n\n- ");
  // Remaining inline " • " between items
  s = s.replace(/\s+\u2022\s+/g, "\n- ");
  // Line-start bullets
  s = s.replace(/^(\s*)\u2022\s+/gm, "$1- ");
  return s;
}

/**
 * Renders assistant copy (interrupt prompts, agenda) with GitHub-flavored Markdown.
 */
export function MarkdownMessage({ children, isDark }: Props) {
  const link = isDark ? "text-blue-400 underline-offset-2 hover:underline" : "text-blue-600 underline-offset-2 hover:underline";
  const codeInline =
    isDark
      ? "rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-[0.9em] text-zinc-200"
      : "rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[0.9em] text-zinc-800";
  const preBlock =
    isDark
      ? "mb-3 overflow-x-auto rounded-lg border border-zinc-700 bg-zinc-950 p-3 font-mono text-[0.85em] text-zinc-200 last:mb-0"
      : "mb-3 overflow-x-auto rounded-lg border border-zinc-200 bg-zinc-50 p-3 font-mono text-[0.85em] text-zinc-800 last:mb-0";

  const components: Components = {
    p: ({ node: _n, ...props }) => (
      <p className="mb-3 last:mb-0 [&:first-child]:mt-0" {...props} />
    ),
    ul: ({ node: _n, ...props }) => (
      <ul className="mb-3 ml-4 list-disc space-y-1 [&>li]:pl-0.5" {...props} />
    ),
    ol: ({ node: _n, ...props }) => (
      <ol className="mb-3 ml-4 list-decimal space-y-1 [&>li]:pl-0.5" {...props} />
    ),
    li: ({ node: _n, ...props }) => <li className="leading-relaxed" {...props} />,
    strong: ({ node: _n, ...props }) => (
      <strong className="font-semibold text-inherit" {...props} />
    ),
    em: ({ node: _n, ...props }) => <em className="italic" {...props} />,
    a: ({ node: _n, ...props }) => (
      <a className={link} target="_blank" rel="noreferrer noopener" {...props} />
    ),
    code: ({ node: _n, className, children, ...props }) => {
      const isBlock = Boolean(className?.includes("language-"));
      if (isBlock) {
        return (
          <code className={`block whitespace-pre ${className ?? ""}`} {...props}>
            {children}
          </code>
        );
      }
      return (
        <code className={codeInline} {...props}>
          {children}
        </code>
      );
    },
    pre: ({ node: _n, children, ...props }) => (
      <pre className={preBlock} {...props}>
        {children}
      </pre>
    ),
    blockquote: ({ node: _n, ...props }) => (
      <blockquote
        className={
          isDark
            ? "mb-3 border-l-2 border-zinc-600 pl-3 text-zinc-400"
            : "mb-3 border-l-2 border-zinc-300 pl-3 text-zinc-600"
        }
        {...props}
      />
    ),
    h1: ({ node: _n, ...props }) => (
      <h1 className="mb-2 mt-4 text-lg font-semibold first:mt-0" {...props} />
    ),
    h2: ({ node: _n, ...props }) => (
      <h2 className="mb-2 mt-3 text-base font-semibold first:mt-0" {...props} />
    ),
    h3: ({ node: _n, ...props }) => (
      <h3 className="mb-2 mt-3 text-sm font-semibold first:mt-0" {...props} />
    ),
    hr: ({ node: _n, ...props }) => (
      <hr
        className={isDark ? "my-4 border-zinc-700" : "my-4 border-zinc-200"}
        {...props}
      />
    ),
  };

  const markdown =
    BULLET.test(children) ? normalizeUnicodeBulletsToMarkdown(children) : children;

  return (
    <div className="markdown-body text-sm leading-relaxed">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
