import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const CodeBlock = ({ language, value, children }) => {
  const [copied, setCopied] = useState(false);
  const codeString = String(children).replace(/\n$/, '');
  const handleCopy = () => {
    navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };
  return (
    <div style={{
      position: 'relative',
      borderRadius: 10,
      margin: '12px 0',
      padding: 2,
      background: 'linear-gradient(90deg, #4fd1c5 0%, #232b3a 100%)',
      boxShadow: '0 2px 12px 0 #10141f44',
      overflow: 'hidden',
      border: 'none',
      display: 'flex',
      alignItems: 'stretch',
    }}>
      <div style={{
        background: '#181f2a',
        borderRadius: 8,
        width: '100%',
        position: 'relative',
        paddingTop: 0,
        paddingBottom: 0,
      }}>
        <button
          onClick={handleCopy}
          style={{
            position: 'absolute',
            top: 8,
            right: 12,
            zIndex: 2,
            background: copied ? '#4fd1c5' : '#232b3a',
            color: copied ? '#232b3a' : '#4fd1c5',
            border: 'none',
            borderRadius: 6,
            padding: '2px 10px',
            fontSize: 13,
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.2s',
          }}
          title={copied ? 'Copied!' : 'Copy code'}
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
        <SyntaxHighlighter
          style={atomDark}
          language={language || 'python'}
          PreTag="div"
          customStyle={{ borderRadius: 8, padding: 16, fontSize: 15, margin: 0, background: 'transparent' }}
        >
          {codeString}
        </SyntaxHighlighter>
      </div>
    </div>
  );
};

const MarkdownRenderer = ({ content }) => (
  <ReactMarkdown
    children={content}
    components={{
      code({ node, inline, className, children, ...props }) {
        const match = /language-(\w+)/.exec(className || '');
        return !inline ? (
          <CodeBlock language={match ? match[1] : 'python'} value={children} {...props}>{children}</CodeBlock>
        ) : (
          <code className={className} {...props}>
            {children}
          </code>
        );
      }
    }}
  />
);

export default MarkdownRenderer;
