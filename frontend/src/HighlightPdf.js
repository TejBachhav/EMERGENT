import React, { useRef, useEffect, useState } from 'react';
import { getDocument } from 'pdfjs-dist/build/pdf';
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.entry';

const HighlightPdf = ({ fileUrl, highlights = [] }) => {
  const iframeRef = useRef(null);
  const [iframeUrl, setIframeUrl] = useState(fileUrl);
  const [isLoaded, setIsLoaded] = useState(false);
  const selectedHighlight = highlights && highlights.length > 0 ? highlights[0] : null;
  const hasCodeSnippet = selectedHighlight && selectedHighlight.code;

  // Find the best search term (longest line)
  const getBestSearchTerm = (codeSnippet) => {
    if (!codeSnippet) return '';
    const lines = codeSnippet.split('\n').map(line => line.trim()).filter(Boolean);
    if (!lines.length) return '';
    lines.sort((a, b) => b.length - a.length);
    return lines[0];
  };

  // On highlight change, use PDF.js to find the page and update iframe src
  useEffect(() => {
    if (!fileUrl || !hasCodeSnippet) return;
    const searchTerm = getBestSearchTerm(selectedHighlight.code);
    let cancelled = false;
    (async () => {
      try {
        const loadingTask = getDocument(fileUrl);
        const pdf = await loadingTask.promise;
        for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
          const page = await pdf.getPage(pageNum);
          const textContent = await page.getTextContent();
          const pageText = textContent.items.map(item => item.str).join(' ');
          if (pageText.replace(/\s+/g, '').toLowerCase().includes(searchTerm.replace(/\s+/g, '').toLowerCase())) {
            if (!cancelled) setIframeUrl(fileUrl + `#page=${pageNum}`);
            break;
          }
        }
      } catch (err) {
        setIframeUrl(fileUrl); // fallback
      }
    })();
    return () => { cancelled = true; };
  }, [fileUrl, hasCodeSnippet, selectedHighlight]);

  return (
    <div style={{ width: '100%', height: 600, position: 'relative', overflow: 'hidden', borderRadius: 8, border: hasCodeSnippet ? '2px solid #4fd1c5' : '1px solid #ccc', background: '#fff' }}>
      <iframe 
        ref={iframeRef}
        src={iframeUrl}
        onLoad={() => setIsLoaded(true)}
        title="PDF Preview"
        style={{ 
          width: '100%', 
          height: '100%',
          border: 'none',
          borderRadius: '8px',
          background: '#fff'
        }}
      />
    </div>
  );
};

export default HighlightPdf;
