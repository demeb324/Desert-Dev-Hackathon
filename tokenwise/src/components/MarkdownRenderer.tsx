import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

// Custom component renderers for OptiGreen theme
const markdownComponents: Components = {
  // Headings
  h1: ({node, ...props}) => (
    <h1 className="text-2xl font-bold text-green-300 mt-4 mb-2 first:mt-0" {...props} />
  ),
  h2: ({node, ...props}) => (
    <h2 className="text-xl font-bold text-green-400 mt-3 mb-2 first:mt-0" {...props} />
  ),
  h3: ({node, ...props}) => (
    <h3 className="text-lg font-semibold text-green-400 mt-2 mb-1 first:mt-0" {...props} />
  ),
  h4: ({node, ...props}) => (
    <h4 className="text-base font-semibold text-green-500 mt-2 mb-1 first:mt-0" {...props} />
  ),
  h5: ({node, ...props}) => (
    <h5 className="text-sm font-semibold text-green-500 mt-2 mb-1 first:mt-0" {...props} />
  ),
  h6: ({node, ...props}) => (
    <h6 className="text-xs font-semibold text-green-500 mt-2 mb-1 first:mt-0" {...props} />
  ),

  // Paragraphs
  p: ({node, ...props}) => (
    <p className="text-gray-300 mb-3 leading-relaxed last:mb-0" {...props} />
  ),

  // Lists
  ul: ({node, ...props}) => (
    <ul className="list-disc list-inside text-gray-300 mb-3 space-y-1" {...props} />
  ),
  ol: ({node, ...props}) => (
    <ol className="list-decimal list-inside text-gray-300 mb-3 space-y-1" {...props} />
  ),
  li: ({node, ...props}) => (
    <li className="text-gray-300 ml-2" {...props} />
  ),

  // Code blocks and inline code
  code: ({node, inline, className, children, ...props}) => {
    if (inline) {
      return (
        <code
          className="bg-gray-800 text-cyan-400 px-1.5 py-0.5 rounded text-xs font-mono"
          {...props}
        >
          {children}
        </code>
      );
    }
    return (
      <code
        className="block bg-gray-800 text-cyan-300 p-3 rounded-lg text-xs font-mono overflow-x-auto whitespace-pre"
        {...props}
      >
        {children}
      </code>
    );
  },
  pre: ({node, ...props}) => (
    <pre className="bg-gray-800 rounded-lg overflow-hidden mb-3 max-w-full" {...props} />
  ),

  // Blockquotes
  blockquote: ({node, ...props}) => (
    <blockquote
      className="border-l-4 border-green-500 pl-4 py-2 my-3 text-gray-400 italic bg-gray-800/30 rounded-r"
      {...props}
    />
  ),

  // Links
  a: ({node, href, ...props}) => (
    <a
      className="text-cyan-400 hover:text-cyan-300 underline transition-colors"
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    />
  ),

  // Horizontal rules
  hr: ({node, ...props}) => (
    <hr className="border-gray-700 my-4" {...props} />
  ),

  // Tables (GitHub Flavored Markdown)
  table: ({node, ...props}) => (
    <div className="overflow-x-auto mb-3">
      <table className="w-full border-collapse border border-gray-700" {...props} />
    </div>
  ),
  thead: ({node, ...props}) => (
    <thead className="bg-gray-800" {...props} />
  ),
  th: ({node, ...props}) => (
    <th
      className="border border-gray-700 px-3 py-2 text-green-400 font-semibold text-left"
      {...props}
    />
  ),
  td: ({node, ...props}) => (
    <td className="border border-gray-700 px-3 py-2 text-gray-300" {...props} />
  ),

  // Strong/Bold
  strong: ({node, ...props}) => (
    <strong className="font-bold text-green-300" {...props} />
  ),

  // Emphasis/Italic
  em: ({node, ...props}) => (
    <em className="italic text-gray-400" {...props} />
  ),

  // Strikethrough (GFM)
  del: ({node, ...props}) => (
    <del className="line-through text-gray-500" {...props} />
  ),

  // Images
  img: ({node, src, alt, ...props}) => (
    <img
      className="max-w-full h-auto rounded-lg my-3"
      src={src}
      alt={alt || ''}
      loading="lazy"
      {...props}
    />
  ),
};

// Safe link URI transformation
const urlTransform = (uri: string): string => {
  // Only allow http/https/mailto protocols
  if (uri.startsWith('http://') || uri.startsWith('https://') || uri.startsWith('mailto:')) {
    return uri;
  }
  // Block everything else (javascript:, data:, etc.)
  return '#';
};

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  className = ''
}) => {
  return (
    <div className={`markdown-content text-left ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={markdownComponents}
        urlTransform={urlTransform}
        disallowedElements={['script', 'iframe', 'object', 'embed']}
        unwrapDisallowed={true}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};
