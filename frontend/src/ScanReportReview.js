import React, { useState, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useNavigate } from 'react-router-dom';
import './App.css';
import MarkdownRenderer from './MarkdownRenderer';

const ScanReportReview = (props) => {
  const token = props.token || localStorage.getItem('token') || '';
  const selectedConversation = props.selectedConversation;
  const setSelectedConversation = props.setSelectedConversation || (() => {});

  const [scanFindings, setScanFindings] = useState([]);
  const [selectedFinding, setSelectedFinding] = useState(null);
  const [pdfBlobUrl, setPdfBlobUrl] = useState(null);
  const [pdfText, setPdfText] = useState('');
  const scanFileInputRef = useRef(null);
  const [scanUploadLoading, setScanUploadLoading] = useState(false);
  const [remediationResult, setRemediationResult] = useState(null);
  const [pdfPreviewError, setPdfPreviewError] = useState(false);
  const navigate = useNavigate();

  // Upload Checkmarx scan report (JSON or PDF)
  const handleUploadScanReport = () => {
    scanFileInputRef.current?.click();
  };

  // Fetch findings for the selected conversation (no PDF fetch)
  const fetchFindings = async (convId) => {
    if (!convId) return;
    fetch(`${process.env.REACT_APP_BACKEND_URL}/scan_findings?conversation_id=${convId}`, {
      headers: { 'Authorization': `Bearer ${token}` },
      credentials: 'include',
    })
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data.findings)) {
          setScanFindings(data.findings);
        }
      });
  };

  const onScanFileInputChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setScanUploadLoading(true);
    setRemediationResult(null);
    setPdfPreviewError(false);
    setPdfBlobUrl(null);
    setPdfText('');
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conversation_id', selectedConversation || '');
    try {
      const res = await fetch(`${process.env.REACT_APP_BACKEND_URL}/upload_scan_report`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        credentials: 'include',
        body: formData
      });
      const data = await res.json();
      if (res.ok && (data.findings || data.pdf_report)) {
        setScanFindings(data.findings || []);
        setPdfText(data.pdf_text || '');
        // Store PDF in frontend for preview
        if (file.type === 'application/pdf') {
          const url = window.URL.createObjectURL(file);
          setPdfBlobUrl(url);
        } else {
          setPdfBlobUrl(null);
        }
      } else {
        alert(data.error || 'Failed to upload scan report.');
      }
    } catch (err) {
      alert('Network error uploading scan report.');
    }
    setScanUploadLoading(false);
  };

  // Interactive remediation actions
  const handleExplainFinding = async (finding) => {
    setRemediationResult(null);
    try {
      const res = await fetch(`${process.env.REACT_APP_BACKEND_URL}/scan_finding/${finding.finding_id}/explain`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        credentials: 'include',
      });
      const data = await res.json();
      if (res.ok && data.explanation) setRemediationResult({ type: 'explanation', text: data.explanation });
      else alert(data.error || 'Failed to get explanation.');
    } catch { alert('Network error.'); }
  };
  const handlePatchFinding = async (finding) => {
    setRemediationResult(null);
    try {
      const res = await fetch(`${process.env.REACT_APP_BACKEND_URL}/scan_finding/${finding.finding_id}/patch`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        credentials: 'include',
      });
      const data = await res.json();
      if (res.ok && data.patch) setRemediationResult({ type: 'patch', text: data.patch });
      else alert(data.error || 'Failed to get patch.');
    } catch { alert('Network error.'); }
  };

  // Secure PDF download (uses blob URL if available)
  const handleDownloadPdf = async () => {
    if (!pdfBlobUrl) return;
    const a = document.createElement('a');
    a.href = pdfBlobUrl;
    a.download = 'checkmarx_report.pdf';
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  React.useEffect(() => {
    if (selectedConversation) {
      fetchFindings(selectedConversation);
    }
    // eslint-disable-next-line
  }, [selectedConversation, token]);

  // Debug: log selected finding when it changes
  React.useEffect(() => {
    if (selectedFinding) {
      console.log('Selected finding:', selectedFinding);
      console.log('Line number type:', typeof selectedFinding.line, 'Value:', selectedFinding.line);
    }
  }, [selectedFinding]);

  // Debug: log findings
  React.useEffect(() => {
    if (scanFindings) {
      console.log('Scan findings:', scanFindings);
    }
  }, [scanFindings]);

  // Only show findings from the most recent extraction
  const getLatestFindings = () => {
    if (!scanFindings.length) return [];
    // Prefer is_latest if present, else use latest created_at
    if (scanFindings.some(f => f.is_latest)) {
      return scanFindings.filter(f => f.is_latest);
    }
    const latestTime = Math.max(...scanFindings.map(f => new Date(f.created_at).getTime()));
    return scanFindings.filter(f => new Date(f.created_at).getTime() === latestTime);
  };
  const latestFindings = getLatestFindings();

  return (
    <div style={{ minHeight: '100vh', width: '100vw', background: 'linear-gradient(135deg, #10141f 0%, #232b3a 100%)', padding: 0, margin: 0, position: 'relative' }}>
      <div style={{ maxWidth: 1800, margin: '0 auto', padding: '48px 8px 40px 8px', boxShadow: '0 8px 48px 0 rgba(79,209,197,0.10), 0 1.5px 0 0 #232b3a', borderRadius: 32, background: 'rgba(16,20,31,0.92)', backdropFilter: 'blur(2px)', position: 'relative', transition: 'box-shadow 0.3s' }}>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 36 }}>
          <h2 style={{ color: '#4fd1c5', fontWeight: 900, fontSize: 38, margin: 0, flex: 1, letterSpacing: 1.2, textShadow: '0 2px 12px #10141f' }}>🛡️ Scan Report Review & Interactive Remediation</h2>
          <button onClick={() => navigate('/')} style={{ background: 'linear-gradient(90deg, #232b3a 60%, #4fd1c5 100%)', color: '#fff', border: 'none', borderRadius: 12, padding: '12px 28px', fontWeight: 700, fontSize: 19, cursor: 'pointer', marginLeft: 28, boxShadow: '0 2px 12px rgba(79,209,197,0.13)' }}>← Back to Chat</button>
        </div>
        <button
          onClick={handleUploadScanReport}
          disabled={scanUploadLoading}
          style={{ background: 'linear-gradient(90deg, #4fd1c5 60%, #232b3a 100%)', color: '#10141f', border: 'none', borderRadius: 12, padding: '14px 36px', fontWeight: 800, fontSize: 20, cursor: 'pointer', marginBottom: 32, boxShadow: '0 2px 12px 0 rgba(79,209,197,0.13)', transition: 'background 0.2s, box-shadow 0.2s', outline: 'none' }}
        >{scanUploadLoading ? 'Uploading...' : '⬆️ Upload Checkmarx Report (JSON/PDF)'}</button>
        <input
          type="file"
          ref={scanFileInputRef}
          style={{ display: 'none' }}
          onChange={onScanFileInputChange}
          accept=".json,.pdf"
        />
        <div style={{ display: 'flex', gap: 12, marginTop: 36, alignItems: 'flex-start', flexWrap: 'nowrap', justifyContent: 'center' }}>
          {/* PDF Preview Column */}
          <div style={{ flex: 1, minWidth: 420, maxWidth: 650, background: 'rgba(24,31,42,0.99)', borderRadius: 22, boxShadow: '0 8px 32px 0 rgba(79,209,197,0.13)', padding: 0, border: '2px solid #232b3a', position: 'relative', marginRight: 0, transition: 'box-shadow 0.2s, border 0.2s' }}>
            {pdfBlobUrl && !pdfPreviewError ? (
              <div style={{padding: 24}}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                  <button 
                    onClick={handleDownloadPdf} 
                    style={{ 
                      color: '#4fd1c5', 
                      fontWeight: 700, 
                      background: 'none', 
                      border: 'none', 
                      cursor: 'pointer', 
                      textDecoration: 'underline', 
                      fontSize: 17 
                    }}
                  >
                    Download Original PDF Report
                  </button>
                  {selectedFinding && (
                    <div style={{ 
                      color: '#ffd166', 
                      fontSize: 15, 
                      padding: '5px 14px',
                      background: 'rgba(35, 43, 58, 0.85)',
                      borderRadius: '6px',
                      fontWeight: 600,
                      border: '1px solid #ffd166',
                      boxShadow: '0 1px 4px 0 rgba(255,209,102,0.08)'
                    }}>
                      {selectedFinding.line ? `Line ${selectedFinding.line} selected` : 'Code snippet selected'}
                    </div>
                  )}
                </div>
                {/* PDF Preview: Browser native viewer (iframe) */}
                <div style={{ marginBottom: 18 }}>
                  <button
                    onClick={() => window.open(pdfBlobUrl, '_blank')}
                    style={{ background: 'linear-gradient(90deg, #4fd1c5 60%, #232b3a 100%)', color: '#10141f', border: 'none', borderRadius: 8, padding: '10px 22px', fontWeight: 700, fontSize: 16, cursor: 'pointer', marginBottom: 8, boxShadow: '0 2px 8px 0 rgba(79,209,197,0.10)' }}
                    disabled={!pdfBlobUrl}
                  >Open PDF in New Tab (Browser Viewer)</button>
                </div>
                {pdfBlobUrl && (
                  <iframe
                    src={pdfBlobUrl}
                    title="PDF Preview"
                    style={{ width: '100%', minHeight: 600, border: '1.5px solid #4fd1c5', borderRadius: 10, background: '#fff', boxShadow: '0 2px 8px rgba(0,0,0,0.10)' }}
                    frameBorder="0"
                  />
                )}
              </div>
            ) : (
              <div style={{ color: '#b3b3d1', fontSize: 18, marginTop: 40, textAlign: 'center' }}>Upload a Checkmarx PDF to preview it here.</div>
            )}
          </div>
          {/* Divider */}
          <div style={{ width: 3, minHeight: 800, background: 'linear-gradient(180deg, #232b3a 0%, #4fd1c5 100%)', borderRadius: 2, opacity: 0.16, alignSelf: 'stretch', margin: '0 0px' }} />
          {/* Findings Column */}
          <div style={{ flex: 1, minWidth: 420, maxWidth: 650, background: 'rgba(24,31,42,0.99)', borderRadius: 22, boxShadow: '0 8px 32px 0 rgba(79,209,197,0.13)', padding: 0, border: '2px solid #232b3a', position: 'relative', marginRight: 0, transition: 'box-shadow 0.2s, border 0.2s' }}>
            <div style={{padding: 24}}>
              <div style={{ fontWeight: 900, color: '#4fd1c5', marginBottom: 16, fontSize: 23, letterSpacing: 0.7, textShadow: '0 1px 8px #10141f' }}>Vulnerable Code Snippets</div>
              <div style={{ maxHeight: 1200, overflowY: 'auto', border: '2px solid #4fd1c5', borderRadius: 12, background: '#181f2a', padding: 18, boxShadow: '0 2px 12px 0 rgba(79,209,197,0.06)' }}>
                {latestFindings.filter(finding => finding.code_snippet && finding.code_snippet.trim()).map(finding => (
                  <div
                    key={finding.finding_id}
                    style={{ marginBottom: 22, padding: 14, borderRadius: 10, background: selectedFinding && selectedFinding.finding_id === finding.finding_id ? 'linear-gradient(90deg, #232b3a 60%, #4fd1c5 100%)' : 'transparent', cursor: 'pointer', border: selectedFinding && selectedFinding.finding_id === finding.finding_id ? '2.5px solid #4fd1c5' : '1.5px solid #232b3a', boxShadow: selectedFinding && selectedFinding.finding_id === finding.finding_id ? '0 2px 12px 0 rgba(79,209,197,0.13)' : 'none', transition: 'background 0.2s, border 0.2s, box-shadow 0.2s' }}
                    onClick={() => { setSelectedFinding(finding); setRemediationResult(null); }}
                    title={`Click to review or remediate`}
                  >
                    <div style={{ fontWeight: 700, color: '#ff5c5c', marginBottom: 6, fontSize: 17 }}>{finding.vuln_name} ({finding.severity})</div>
                    <div style={{ color: '#b3b3d1', fontSize: 14, marginBottom: 5 }}>{finding.file_path}:{finding.line}</div>
                    <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 7, padding: 10, fontSize: 15, margin: 0 }}>{finding.code_snippet}</SyntaxHighlighter>
                  </div>
                ))}
                {latestFindings.filter(finding => finding.code_snippet && finding.code_snippet.trim()).length === 0 && latestFindings.length > 0 && (
                  <div style={{ color: '#ffd166', fontSize: 15, marginBottom: 14 }}>
                    No code snippets extracted. Raw findings:<br />
                    <pre style={{ color: '#fff', background: '#232b3a', borderRadius: 7, padding: 10, fontSize: 13, maxHeight: 220, overflow: 'auto' }}>{JSON.stringify(latestFindings, null, 2)}</pre>
                  </div>
                )}
              </div>
            </div>
          </div>
          {/* Divider */}
          <div style={{ width: 3, minHeight: 800, background: 'linear-gradient(180deg, #232b3a 0%, #4fd1c5 100%)', borderRadius: 2, opacity: 0.16, alignSelf: 'stretch', margin: '0 0px' }} />
          {/* Remediation Column */}
          <div style={{ flex: 1, minWidth: 420, maxWidth: 650, background: 'rgba(24,31,42,0.99)', borderRadius: 22, boxShadow: '0 8px 32px 0 rgba(79,209,197,0.13)', padding: 0, border: '2px solid #232b3a', position: 'relative', transition: 'box-shadow 0.2s, border 0.2s' }}>
            <div style={{padding: 24}}>
              {selectedFinding ? (
                <div style={{ border: '2px solid #4fd1c5', borderRadius: 12, background: '#181f2a', padding: 28, boxShadow: '0 2px 12px 0 rgba(79,209,197,0.06)' }}>
                  <div style={{ fontWeight: 900, color: '#4fd1c5', marginBottom: 14, fontSize: 20, textShadow: '0 1px 8px #10141f' }}>Remediation for: {selectedFinding.vuln_name}</div>
                  <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 7, padding: 10, fontSize: 16, marginBottom: 18 }}>{selectedFinding.code_snippet}</SyntaxHighlighter>
                  <div style={{ display: 'flex', gap: 18, marginBottom: 18 }}>
                    <button onClick={() => handleExplainFinding(selectedFinding)} style={{ background: 'linear-gradient(90deg, #ffd166 60%, #232b3a 100%)', color: '#232b3a', border: 'none', borderRadius: 9, padding: '12px 22px', fontWeight: 700, fontSize: 16, cursor: 'pointer', boxShadow: '0 2px 12px 0 rgba(255,209,102,0.13)' }}>Explain Vulnerability & Impact</button>
                    <button onClick={() => handlePatchFinding(selectedFinding)} style={{ background: 'linear-gradient(90deg, #4fd1c5 60%, #232b3a 100%)', color: '#10141f', border: 'none', borderRadius: 9, padding: '12px 22px', fontWeight: 700, fontSize: 16, cursor: 'pointer', boxShadow: '0 2px 12px 0 rgba(79,209,197,0.13)' }}>Patch Recommendation</button>
                  </div>
                  {remediationResult && (
                    <div style={{ background: remediationResult.type === 'patch' ? '#232b3a' : '#232b3a', color: '#fff', borderRadius: 7, padding: 16, fontSize: 16, border: '1.5px solid #4fd1c5', marginTop: 10 }}>
                      <b>{remediationResult.type === 'patch' ? 'Patch Recommendation:' : 'Explanation:'}</b>
                      <div style={{ marginTop: 10 }}>
                        <MarkdownRenderer content={remediationResult.text} />
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ color: '#b3b3d1', fontSize: 17, marginTop: 40, textAlign: 'center' }}>Select a code snippet to review or remediate.</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ScanReportReview;
