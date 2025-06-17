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
      h1: ({node, ...props}) => <h1 style={{ color: '#4fd1c5', fontSize: 28, fontWeight: 900, margin: '18px 0 10px 0', borderBottom: '2px solid #4fd1c5', paddingBottom: 4, letterSpacing: 0.5 }} {...props} />,
      h2: ({node, ...props}) => <h2 style={{ color: '#4fd1c5', fontSize: 22, fontWeight: 800, margin: '16px 0 8px 0', borderBottom: '1.5px solid #4fd1c5', paddingBottom: 2, letterSpacing: 0.3 }} {...props} />,
      h3: ({node, ...props}) => <h3 style={{ color: '#ffd166', fontSize: 18, fontWeight: 700, margin: '14px 0 6px 0' }} {...props} />,
      ul: ({node, ...props}) => <ul style={{ paddingLeft: 24, margin: '10px 0', color: '#fff', fontSize: 15, lineHeight: 1.7 }} {...props} />,
      ol: ({node, ...props}) => <ol style={{ paddingLeft: 24, margin: '10px 0', color: '#fff', fontSize: 15, lineHeight: 1.7 }} {...props} />,
      li: ({node, ...props}) => <li style={{ margin: '6px 0', fontSize: 15, lineHeight: 1.7 }} {...props} />,
      blockquote: ({node, ...props}) => <blockquote style={{ borderLeft: '4px solid #4fd1c5', background: '#232b3a', color: '#b3b3d1', margin: '14px 0', padding: '10px 18px', borderRadius: 8, fontStyle: 'italic', fontSize: 15 }} {...props} />,
      table: ({node, ...props}) => <table style={{ width: '100%', borderCollapse: 'collapse', margin: '16px 0', background: '#181f2a', color: '#fff', fontSize: 15, borderRadius: 8, overflow: 'hidden' }} {...props} />,
      th: ({node, ...props}) => <th style={{ border: '1px solid #4fd1c5', padding: '8px 12px', background: '#232b3a', color: '#4fd1c5', fontWeight: 700 }} {...props} />,
      td: ({node, ...props}) => <td style={{ border: '1px solid #232b3a', padding: '8px 12px', background: '#181f2a' }} {...props} />,
      code({ node, inline, className, children, ...props }) {
        const match = /language-(\w+)/.exec(className || '');
        return !inline ? (
          <CodeBlock language={match ? match[1] : 'python'} value={children} {...props}>{children}</CodeBlock>
        ) : (
          <code className={className} style={{ background: '#232b3a', color: '#ffd166', borderRadius: 5, padding: '2px 6px', fontSize: 14, fontWeight: 600 }} {...props}>
            {children}
          </code>
        );
      }
    }}
  />
);

export default MarkdownRenderer;
