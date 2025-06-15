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

  // Debug: log findings
  React.useEffect(() => {
    if (scanFindings) {
      console.log('Scan findings:', scanFindings);
    }
  }, [scanFindings]);

  return (
    <div style={{ padding: 32, background: '#10141f', minHeight: '100vh' }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 18 }}>
        <h2 style={{ color: '#4fd1c5', fontWeight: 800, fontSize: 28, margin: 0, flex: 1 }}>Scan Report Review & Interactive Remediation</h2>
        <button onClick={() => navigate('/')} style={{ background: '#232b3a', color: '#fff', border: 'none', borderRadius: 8, padding: '8px 18px', fontWeight: 600, fontSize: 15, cursor: 'pointer', marginLeft: 16 }}>← Back to Chat</button>
      </div>
      <button
        onClick={handleUploadScanReport}
        disabled={scanUploadLoading}
        style={{ background: '#232b3a', color: '#4fd1c5', border: 'none', borderRadius: 8, padding: '8px 18px', fontWeight: 600, fontSize: 15, cursor: 'pointer', marginBottom: 18 }}
      >{scanUploadLoading ? 'Uploading...' : '⬆️ Upload Checkmarx Report (JSON/PDF)'}</button>
      <input
        type="file"
        ref={scanFileInputRef}
        style={{ display: 'none' }}
        onChange={onScanFileInputChange}
        accept=".json,.pdf"
      />
      <div style={{ display: 'flex', gap: 32, marginTop: 24, alignItems: 'flex-start' }}>
        {/* Left: PDF Preview */}
        <div style={{ flex: 1, minWidth: 350, maxWidth: 600 }}>
          {pdfBlobUrl && !pdfPreviewError ? (
            <div>
              <button onClick={handleDownloadPdf} style={{ color: '#4fd1c5', fontWeight: 600, background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline', fontSize: 16, marginBottom: 8 }}>Download Original PDF Report</button>
              <iframe src={pdfBlobUrl} title="Checkmarx PDF Preview" style={{ width: '100%', height: 1000, border: '1.5px solid #4fd1c5', borderRadius: 8, marginTop: 8, background: '#fff' }}></iframe>
            </div>
          ) : (
            <div style={{ color: '#b3b3d1', fontSize: 15, marginTop: 32 }}>Upload a Checkmarx PDF to preview it here.</div>
          )}
        </div>
        {/* Right: Findings and Remediation */}
        <div style={{ flex: 1, minWidth: 320 }}>
          {scanFindings.length > 0 ? (
            <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
              <div style={{ flex: 1, minWidth: 320 }}>
                <div style={{ fontWeight: 700, color: '#4fd1c5', marginBottom: 8 }}>Vulnerable Code Snippets</div>
                <div style={{ maxHeight: 1000, overflowY: 'auto', border: '1.5px solid #4fd1c5', borderRadius: 8, background: '#181f2a', padding: 12 }}>
                  {scanFindings.filter(finding => finding.code_snippet && finding.code_snippet.trim()).map(finding => (
                    <div
                      key={finding.finding_id}
                      style={{ marginBottom: 18, padding: 8, borderRadius: 6, background: selectedFinding && selectedFinding.finding_id === finding.finding_id ? '#232b3a' : 'transparent', cursor: 'pointer', border: selectedFinding && selectedFinding.finding_id === finding.finding_id ? '1.5px solid #4fd1c5' : '1px solid #232b3a' }}
                      onClick={() => { setSelectedFinding(finding); setRemediationResult(null); }}
                      title={`Click to review or remediate`}
                    >
                      <div style={{ fontWeight: 600, color: '#ff5c5c', marginBottom: 4 }}>{finding.vuln_name} ({finding.severity})</div>
                      <div style={{ color: '#b3b3d1', fontSize: 13, marginBottom: 4 }}>{finding.file_path}:{finding.line}</div>
                      <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 6, padding: 8, fontSize: 14, margin: 0 }}>{finding.code_snippet}</SyntaxHighlighter>
                    </div>
                  ))}
                  {scanFindings.filter(finding => finding.code_snippet && finding.code_snippet.trim()).length === 0 && scanFindings.length > 0 && (
                    <div style={{ color: '#ffd166', fontSize: 14, marginBottom: 12 }}>
                      No code snippets extracted. Raw findings:<br />
                      <pre style={{ color: '#fff', background: '#232b3a', borderRadius: 6, padding: 8, fontSize: 12, maxHeight: 200, overflow: 'auto' }}>{JSON.stringify(scanFindings, null, 2)}</pre>
                    </div>
                  )}
                </div>
              </div>
              <div style={{ flex: 1, minWidth: 320 }}>
                {selectedFinding ? (
                  <div style={{ border: '1.5px solid #4fd1c5', borderRadius: 8, background: '#181f2a', padding: 18 }}>
                    <div style={{ fontWeight: 700, color: '#4fd1c5', marginBottom: 8 }}>Remediation for: {selectedFinding.vuln_name}</div>
                    <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 6, padding: 8, fontSize: 14, marginBottom: 12 }}>{selectedFinding.code_snippet}</SyntaxHighlighter>
                    <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
                      <button onClick={() => handleExplainFinding(selectedFinding)} style={{ background: '#232b3a', color: '#ffd166', border: 'none', borderRadius: 8, padding: '8px 14px', fontWeight: 600, fontSize: 15, cursor: 'pointer' }}>Explain Vulnerability & Impact</button>
                      <button onClick={() => handlePatchFinding(selectedFinding)} style={{ background: '#232b3a', color: '#4fd1c5', border: 'none', borderRadius: 8, padding: '8px 14px', fontWeight: 600, fontSize: 15, cursor: 'pointer' }}>Patch Recommendation</button>
                    </div>
                    {remediationResult && (
                      <div style={{ background: remediationResult.type === 'patch' ? '#232b3a' : '#232b3a', color: '#fff', borderRadius: 6, padding: 12, fontSize: 15, border: '1px solid #4fd1c5' }}>
                        <b>{remediationResult.type === 'patch' ? 'Patch Recommendation:' : 'Explanation:'}</b>
                        <div style={{ marginTop: 6 }}>
                          <MarkdownRenderer content={remediationResult.text} />
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div style={{ color: '#b3b3d1', fontSize: 15, marginTop: 32 }}>Select a code snippet to review or remediate.</div>
                )}
              </div>
            </div>
          ) : (
            <div style={{ color: '#b3b3d1', fontSize: 15, marginTop: 32 }}>Upload a Checkmarx report to see findings here.</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ScanReportReview;
