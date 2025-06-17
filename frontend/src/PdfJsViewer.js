import React, { useEffect, useRef, useState } from 'react';
import { GlobalWorkerOptions, getDocument, renderTextLayer } from 'pdfjs-dist/build/pdf';
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.entry';
import 'pdfjs-dist/web/pdf_viewer.css';

GlobalWorkerOptions.workerSrc = pdfjsWorker;

// This component is now a simple PDF.js-based viewer with no highlighting logic
const PdfJsViewer = ({ fileUrl }) => {
  const containerRef = useRef(null);
  const [numPages, setNumPages] = useState(0);
  const [pdf, setPdf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [scale, setScale] = useState(1.2);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);
    setPdf(null);
    if (!fileUrl) {
      setLoading(false);
      return;
    }
    getDocument(fileUrl).promise
      .then(doc => {
        if (!isMounted) return;
        setPdf(doc);
        setNumPages(doc.numPages);
        setLoading(false);
      })
      .catch(e => {
        if (!isMounted) return;
        setError('Failed to load PDF.');
        setLoading(false);
      });
    return () => { isMounted = false; };
  }, [fileUrl]);

  useEffect(() => {
    if (!pdf || !containerRef.current) return;
    containerRef.current.innerHTML = '';
    (async () => {
      for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
        const pageObj = await pdf.getPage(pageNum);
        const viewport = pageObj.getViewport({ scale });
        const resolutionMultiplier = 2;
        const canvas = document.createElement('canvas');
        canvas.width = viewport.width * resolutionMultiplier;
        canvas.height = viewport.height * resolutionMultiplier;
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;
        const ctx = canvas.getContext('2d');
        const renderContext = {
          canvasContext: ctx,
          viewport: pageObj.getViewport({ scale: scale * resolutionMultiplier })
        };
        await pageObj.render(renderContext).promise;
        const pageDiv = document.createElement('div');
        pageDiv.className = 'pageContainer';
        pageDiv.style.position = 'relative';
        pageDiv.style.border = '1px solid #ddd';
        pageDiv.style.borderRadius = '4px';
        pageDiv.style.overflow = 'hidden';
        pageDiv.style.background = '#fff';
        pageDiv.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
        pageDiv.appendChild(canvas);
        // Render text layer for selectable/searchable text
        const textContent = await pageObj.getTextContent();
        const textLayerDiv = document.createElement('div');
        textLayerDiv.className = 'textLayer';
        textLayerDiv.style.width = canvas.style.width;
        textLayerDiv.style.height = canvas.style.height;
        textLayerDiv.style.position = 'absolute';
        textLayerDiv.style.left = '0';
        textLayerDiv.style.top = '0';
        pageDiv.appendChild(textLayerDiv);
        await renderTextLayer({
            textContent: textContent,
            container: textLayerDiv,
            viewport: viewport,
            enhanceTextSelection: true,
        }).promise;
        containerRef.current.appendChild(pageDiv);
      }
    })();
  }, [pdf, scale]);

  return (
    <div style={{ width: '100%', height: 600, overflowY: 'auto', background: '#f8f9fa', borderRadius: 8, border: '1px solid #ccc', position: 'relative', padding: '8px' }}>
      {loading && <div style={{ padding: 20, color: '#4fd1c5', textAlign: 'center' }}>Loading PDF...</div>}
      {error && <div style={{ padding: 20, color: '#ff5c5c', textAlign: 'center' }}>{error}</div>}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, margin: '0 0 16px 0', padding: '8px 16px', background: '#fff', borderRadius: '6px', border: '1px solid #e0e0e0' }}>
        <button 
          onClick={() => setScale(s => Math.max(0.5, s - 0.2))}
          style={{ background: '#232b3a', color: '#fff', border: 'none', borderRadius: '4px', padding: '6px 12px', cursor: 'pointer' }}
        >-</button>
        <span style={{ color: '#666', fontWeight: '600' }}>Zoom: {Math.round(scale * 100)}%</span>
        <button 
          onClick={() => setScale(s => Math.min(3, s + 0.2))}
          style={{ background: '#232b3a', color: '#fff', border: 'none', borderRadius: '4px', padding: '6px 12px', cursor: 'pointer' }}
        >+</button>
        <span style={{ marginLeft: 16, color: '#888', fontSize: '14px' }}>Pages: {numPages}</span>
      </div>
      <div 
        ref={containerRef} 
        style={{ 
          minHeight: 500, 
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '16px'
        }} 
      />
    </div>
  );
};

export default PdfJsViewer;
