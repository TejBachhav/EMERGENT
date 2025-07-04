import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ReactMarkdown from 'react-markdown';
import { BrowserRouter as Router, Routes, Route, useNavigate, Navigate } from 'react-router-dom';
import './App.css';
import ScanReportReview from './ScanReportReview';

// Utility to generate a UUID (RFC4122 version 4)
function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// Returns a valid, non-empty conversation_id. If invalid, generates a new one and updates state.
function getValidConversationId(selectedConversation, setSelectedConversation) {
  if (typeof selectedConversation === 'string' && selectedConversation.trim() && selectedConversation !== 'None') {
    return selectedConversation;
  }
  const newId = generateUUID();
  setSelectedConversation(newId);
  return newId;
}

const App = ({ token, setToken, selectedConversation, setSelectedConversation, isLoggedIn, setIsLoggedIn, username, setUsername }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [showReactions, setShowReactions] = useState(null);
  const [loginForm, setLoginForm] = useState({ username: '', password: '' });
  const [loginError, setLoginError] = useState('');
  const [showRegister, setShowRegister] = useState(false);
  const [registerForm, setRegisterForm] = useState({ username: '', password: '' });
  const [registerError, setRegisterError] = useState('');
  const [registerSuccess, setRegisterSuccess] = useState('');
  const [conversations, setConversations] = useState([]);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);
  const [copiedCodeIndex, setCopiedCodeIndex] = useState(null);
  const [geminiApiKey, setGeminiApiKey] = useState(() => localStorage.getItem('geminiApiKey') || '');
  const [showGeminiKeyInput, setShowGeminiKeyInput] = useState(false);
  const [showProfileCard, setShowProfileCard] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const inputMenuRef = useRef(null);
  const fileInputRef = useRef(null);

  // Dictation (Speech-to-Text) support
  const [isDictating, setIsDictating] = useState(false);
  const [dictateTarget, setDictateTarget] = useState('input'); // 'input' or message id
  const recognitionRef = useRef(null);

  // Text-to-Speech (TTS) for bot responses
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [speakingMsgId, setSpeakingMsgId] = useState(null);
  const utterRef = useRef(null);

  const handleSpeak = (text, msgId) => {
    if (!('speechSynthesis' in window)) {
      alert('Text-to-speech is not supported in this browser.');
      return;
    }
    window.speechSynthesis.cancel(); // Stop any ongoing speech
    const utter = new window.SpeechSynthesisUtterance(text);
    utter.lang = 'en-US';
    utter.rate = 1.0;
    utter.onstart = () => {
      setIsSpeaking(true);
      setIsPaused(false);
      setSpeakingMsgId(msgId);
    };
    utter.onend = () => {
      setIsSpeaking(false);
      setIsPaused(false);
      setSpeakingMsgId(null);
    };
    utter.onerror = () => {
      setIsSpeaking(false);
      setIsPaused(false);
      setSpeakingMsgId(null);
    };
    utterRef.current = utter;
    window.speechSynthesis.speak(utter);
  };

  const handlePause = () => {
    if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {
      window.speechSynthesis.pause();
      setIsPaused(true);
    }
  };

  const handleResume = () => {
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
      setIsPaused(false);
    }
  };

  const handleStop = () => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
    setIsPaused(false);
    setSpeakingMsgId(null);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isLoggedIn && token) {
      // Fetch chat history for the user
      fetch(`${process.env.REACT_APP_BACKEND_URL}/chat_history`, {
        headers: { 'Authorization': `Bearer ${token}` },
        credentials: 'include',
      })
        .then(res => res.json())
        .then(data => {
          if (Array.isArray(data.history)) {
            setMessages(data.history);
          } else {
            setMessages([]);
          }
        })
        .catch(() => setMessages([]));
    }
  }, [isLoggedIn, token]);

  useEffect(() => {
    if (!isLoggedIn) {
      setMessages([]);
    }
  }, [isLoggedIn]);

  // Fetch conversations when logged in
  useEffect(() => {
    if (isLoggedIn && token) {
      fetch(`${process.env.REACT_APP_BACKEND_URL}/conversations`, {
        headers: { 'Authorization': `Bearer ${token}` },
        credentials: 'include',
      })
        .then(res => res.json())
        .then(data => {
          if (Array.isArray(data.conversations)) {
            setConversations(data.conversations);
          } else {
            setConversations([]);
          }
        })
        .catch(() => setConversations([]));
    } else {
      setConversations([]);
    }
  }, [isLoggedIn, token]);

  // Fetch chat history for selected conversation
  useEffect(() => {
    if (isLoggedIn && token && selectedConversation) {
      setConversationLoading(true);
      fetch(`${process.env.REACT_APP_BACKEND_URL}/chat_history?conversation_id=${selectedConversation}`, {
        headers: { 'Authorization': `Bearer ${token}` },
        credentials: 'include',
      })
        .then(res => res.json())
        .then(data => {
          if (Array.isArray(data.history)) {
            setMessages(data.history);
          } else {
            setMessages([]);
          }
          setConversationLoading(false);
        })
        .catch(() => {
          setMessages([]);
          setConversationLoading(false);
        });
    } else if (isLoggedIn && token && selectedConversation === null) {
      // No conversation selected, clear messages
      setMessages([]);
    }
  }, [isLoggedIn, token, selectedConversation]);

  // When logging in, auto-select most recent conversation
  useEffect(() => {
    if (isLoggedIn && conversations.length > 0 && selectedConversation === undefined) {
      setSelectedConversation(conversations[0].conversation_id);
    }
  }, [isLoggedIn, conversations, selectedConversation]);

  const detectCodeBlocks = (text) => {
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)\n```/g;
    const inlineCodeRegex = /`([^`]+)`/g;
    
    return codeBlockRegex.test(text) || inlineCodeRegex.test(text);
  };

  const parseMessageContent = (text) => {
    const parts = [];
    let lastIndex = 0;
    
    // Handle code blocks
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)\n```/g;
    let match;
    
    while ((match = codeBlockRegex.exec(text)) !== null) {
      // Add text before code block
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: text.slice(lastIndex, match.index)
        });
      }
      
      // Add code block
      parts.push({
        type: 'codeblock',
        language: match[1] || 'javascript',
        content: match[2]
      });
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add remaining text
    if (lastIndex < text.length) {
      parts.push({
        type: 'text',
        content: text.slice(lastIndex)
      });
    }
    
    // If no code blocks found, handle inline code
    if (parts.length === 0) {
      const inlineCodeRegex = /`([^`]+)`/g;
      let textContent = text;
      let inlineMatch;
      let processedParts = [];
      lastIndex = 0;
      
      while ((inlineMatch = inlineCodeRegex.exec(text)) !== null) {
        if (inlineMatch.index > lastIndex) {
          processedParts.push({
            type: 'text',
            content: text.slice(lastIndex, inlineMatch.index)
          });
        }
        
        processedParts.push({
          type: 'inline-code',
          content: inlineMatch[1]
        });
        
        lastIndex = inlineMatch.index + inlineMatch[0].length;
      }
      
      if (lastIndex < text.length) {
        processedParts.push({
          type: 'text',
          content: text.slice(lastIndex)
        });
      }
      
      return processedParts.length > 0 ? processedParts : [{ type: 'text', content: text }];
    }
    
    return parts;
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoginError('');
    try {
      const res = await fetch(`${process.env.REACT_APP_BACKEND_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(loginForm),
      });
      const data = await res.json();
      if (res.ok && data.token) {
        setToken(data.token);
        setIsLoggedIn(true);
        setUsername(loginForm.username); // Set username in state
        localStorage.setItem("token", data.token);
        localStorage.setItem("username", loginForm.username);
        setLoginForm({ username: '', password: '' });
      } else {
        setLoginError(data.error || 'Login failed');
      }
    } catch (err) {
      setLoginError('Network error');
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setRegisterError('');
    setRegisterSuccess('');
    try {
      const res = await fetch(`${process.env.REACT_APP_BACKEND_URL}/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(registerForm),
      });
      const data = await res.json();
      if (res.ok) {
        setRegisterSuccess('Registration successful! You can now log in.');
        setRegisterForm({ username: '', password: '' });
        setShowRegister(false);
      } else {
        setRegisterError(data.error || 'Registration failed');
      }
    } catch (err) {
      setRegisterError('Network error');
    }
  };

  const handleStartNewConversation = () => {
    setSelectedConversation(null);
    setMessages([]);
    setInputMessage('');
    setShowSidebar(false);
  };

  const handleSelectConversation = (convId) => {
    setSelectedConversation(convId);
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isSending || !isLoggedIn) return;

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString(),
      hasCode: detectCodeBlocks(inputMessage),
      reactions: []
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsSending(true);
    setIsTyping(true);

    try {
      const conversation_id = getValidConversationId(selectedConversation, setSelectedConversation);
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/stream_chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        credentials: 'include',
        body: JSON.stringify({ message: inputMessage, conversation_id, gemini_api_key: geminiApiKey || undefined }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      // Add bot message
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: data.response,
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        hasCode: detectCodeBlocks(data.response),
        reactions: [],
      }]);
      setIsTyping(false);
      setIsSending(false);
      // If this was a new conversation, update conversation list and select it
      if (!selectedConversation && data.conversation_id) {
        setSelectedConversation(data.conversation_id);
        // Refetch conversations
        fetch(`${process.env.REACT_APP_BACKEND_URL}/conversations`, {
          headers: { 'Authorization': `Bearer ${token}` },
          credentials: 'include',
        })
          .then(res => res.json())
          .then(data => {
            if (Array.isArray(data.conversations)) {
              setConversations(data.conversations);
            }
          });
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: "🚨 **Connection Error**\n\nI'm having trouble connecting to my security database. This could be due to:\n\n• Network connectivity issues\n• Backend service unavailable\n• Authentication problems\n\nPlease try again in a moment. If the issue persists, check your connection or contact support.",
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        isError: true,
        hasCode: false,
        reactions: []
      }]);
      setIsTyping(false);
      setIsSending(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      const newHeight = Math.min(Math.max(textarea.scrollHeight, 35), 120);
      textarea.style.height = `${newHeight}px`;
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputMessage]);

  const handleReaction = (messageId, reaction) => {
    setMessages(prev => prev.map(msg => {
      if (msg.id === messageId) {
        const existingReaction = msg.reactions.find(r => r.emoji === reaction);
        if (existingReaction) {
          return {
            ...msg,
            reactions: msg.reactions.map(r => 
              r.emoji === reaction 
                ? { ...r, count: r.count + 1 }
                : r
            )
          };
        } else {
          return {
            ...msg,
            reactions: [...msg.reactions, { emoji: reaction, count: 1 }]
          };
        }
      }
      return msg;
    }));
    setShowReactions(null);
  };

  const formatMessageText = (content) => {
    // Handle markdown-style formatting
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/^• (.*$)/gim, '<div class="bullet-point">• $1</div>')
      .replace(/^\d+\. (.*$)/gim, '<div class="numbered-point">$&</div>');
  };

  const quickPrompts = [
    "🔍 Analyze this code for vulnerabilities",
    "🚨 Common SQL injection patterns",
    "🔒 Secure authentication best practices",
    "📋 OWASP Top 10 vulnerabilities",
    "🛡️ Input validation techniques"
  ];

  const handleQuickPrompt = (prompt) => {
    setInputMessage(prompt.replace(/^[🔍🚨🔒📋🛡️] /, ''));
  };

  const renderMessageContent = (message, msgIndex) => {
    if (message.sender === 'bot') {
      let text = message.text;
      if (typeof text === 'object' && text !== null && text.response) {
        text = text.response;
      }
      if (typeof text === 'string' && text.trim().startsWith('{') && text.trim().endsWith('}')) {
        try {
          const parsed = JSON.parse(text);
          if (parsed.response) text = parsed.response;
        } catch (e) { /* not JSON, ignore */ }
      }
      return (
        <div style={{ position: 'relative' }}>
          <ReactMarkdown
            children={text}
            components={{
              p({ node, children, ...props }) {
                // Avoid <div> inside <p> hydration errors by using a fragment
                return <>{children}</>;
              },
              ol({node, ...props}) {
                return <ol style={{ paddingLeft: 24, margin: '8px 0', color: '#fff' }} {...props} />;
              },
              ul({node, ...props}) {
                return <ul style={{ paddingLeft: 20, margin: '8px 0', color: '#fff' }} {...props} />;
              },
              li({node, ...props}) {
                return <li style={{ margin: '4px 0', fontSize: 15, lineHeight: 1.7 }} {...props} />;
              },
              code({node, inline, className, children, ...props}) {
                if (!inline) {
                  const codeString = String(children).replace(/\n$/, '');
                  return (
                    <div className="code-block-container enhanced-code-block" style={{ position: 'relative', margin: '12px 0', borderRadius: 12, overflow: 'hidden', border: '1.5px solid #4fd1c5', background: '#181f2a', boxShadow: '0 4px 15px rgba(0,0,0,0.3)', maxHeight: 340, overflowY: 'auto', padding: 0 }}>
                      <div className="code-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 16px', background: 'linear-gradient(135deg, #232b3a 60%, #4fd1c5 100%)', borderBottom: '1px solid #4fd1c5' }}>
                        <span className="code-language" style={{ fontSize: 12, fontWeight: 600, color: '#4fd1c5', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{/language-(\w+)/.exec(className || '')?.[1] || 'text'}</span>
                        <button
                          className="code-action-btn"
                          style={{ background: 'none', border: 'none', color: copiedCodeIndex === msgIndex ? '#4fd1c5' : '#fff', cursor: 'pointer', fontSize: 13, borderRadius: 4, padding: '2px 8px', transition: 'color 0.2s' }}
                          onClick={() => {
                            navigator.clipboard.writeText(codeString);
                            setCopiedCodeIndex(msgIndex);
                            setTimeout(() => setCopiedCodeIndex(null), 1200);
                          }}
                          title="Copy code"
                        >
                          {copiedCodeIndex === msgIndex ? 'Copied!' : '📋 Copy'}
                        </button>
                      </div>
                      <SyntaxHighlighter
                        style={atomDark}
                        language={/language-(\w+)/.exec(className || '')?.[1] || 'text'}
                        PreTag="div"
                        customStyle={{
                          margin: 0,
                          borderRadius: '0 0 8px 8px',
                          background: 'rgba(0, 0, 0, 0.7)',
                          overflowX: 'auto',
                          maxWidth: '100%',
                          fontSize: 15,
                          padding: 18,
                        }}
                        showLineNumbers={true}
                        {...props}
                      >
                        {codeString}
                      </SyntaxHighlighter>
                    </div>
                  );
                } else {
                  return <code className={className} style={{ background: '#232b3a', color: '#4fd1c5', borderRadius: 4, padding: '2px 6px', fontSize: 14 }} {...props}>{children}</code>;
                }
              }
            }}
          />
          {/* Copy and Dictate buttons for each bot message */}
          <div style={{ display: 'flex', gap: 8, position: 'absolute', top: 0, right: 0 }}>
            <button
              style={{ background: 'none', border: 'none', color: '#4fd1c5', fontSize: 20, cursor: 'pointer', padding: 2 }}
              title="Copy response"
              onClick={() => navigator.clipboard.writeText(text)}
            >📋</button>
            <button
              style={{ background: 'none', border: 'none', color: '#4fd1c5', fontSize: 20, cursor: 'pointer', padding: 2 }}
              title="Dictate response (text-to-speech)"
              onClick={() => handleSpeak(text, message.id)}
              disabled={isSpeaking && speakingMsgId !== message.id}
            >🔊</button>
            {isSpeaking && speakingMsgId === message.id && !isPaused && (
              <button
                style={{ background: 'none', border: 'none', color: '#4fd1c5', fontSize: 20, cursor: 'pointer', padding: 2 }}
                title="Pause reading"
                onClick={handlePause}
              >⏸️</button>
            )}
            {isSpeaking && speakingMsgId === message.id && isPaused && (
              <button
                style={{ background: 'none', border: 'none', color: '#4fd1c5', fontSize: 20, cursor: 'pointer', padding: 2 }}
                title="Resume reading"
                onClick={handleResume}
              >▶️</button>
            )}
            {isSpeaking && speakingMsgId === message.id && (
              <button
                style={{ background: 'none', border: 'none', color: '#4fd1c5', fontSize: 20, cursor: 'pointer', padding: 2 }}
                title="Stop reading"
                onClick={handleStop}
              >⏹️</button>
            )}
          </div>
        </div>
      );
    }

    const parts = parseMessageContent(message.text);
    
    return parts.map((part, index) => {
      switch (part.type) {
        case 'codeblock':
          return (
            <div key={index} className="code-block-container" style={{ overflowX: 'auto', overflowY: 'auto', maxWidth: '100%', maxHeight: '300px', wordWrap: 'break-word' }}>
              <div className="code-header">
                <span className="code-language">{part.language}</span>
                <div className="code-actions">
                  <button 
                    className="code-action-btn"
                    onClick={() => navigator.clipboard.writeText(part.content)}
                    title="Copy code"
                  >
                    📋
                  </button>
                  <div className="security-indicator">
                    <span className="security-badge">🛡️ Security</span>
                  </div>
                </div>
              </div>
              <SyntaxHighlighter
                language={part.language}
                style={atomDark}
                customStyle={{
                  margin: 0,
                  borderRadius: '0 0 8px 8px',
                  background: 'rgba(0, 0, 0, 0.7)',
                  overflowX: 'auto',
                  maxWidth: '100%',
                }}
                showLineNumbers={true}
              >
                {part.content}
              </SyntaxHighlighter>
            </div>
          );
        case 'inline-code':
          return (
            <code key={index} className="inline-code" style={{ wordWrap: 'break-word', overflowWrap: 'break-word' }}>               {part.content}
            </code>
          );
        default:
          return (
            <span 
              key={index}
              style={{ wordWrap: 'break-word', overflowWrap: 'break-word' }}
              dangerouslySetInnerHTML={{ 
                __html: formatMessageText(part.content) 
              }}
            />
          );
      }
    });
  };

  const handleGeminiKeySave = () => {
    localStorage.setItem('geminiApiKey', geminiApiKey);
    setShowGeminiKeyInput(false);
  };

  // Dictation logic
  const handleStartDictation = (target = 'input', messageId = null) => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Speech recognition is not supported in this browser.');
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      if (target === 'input') {
        setInputMessage(prev => prev ? prev + ' ' + transcript : transcript);
      } else if (target && typeof target === 'string') {
        setMessages(prevMsgs => prevMsgs.map(m => m.id === target ? { ...m, text: m.text + ' ' + transcript } : m));
      }
      setIsDictating(false);
      setDictateTarget('input');
    };
    recognition.onerror = (event) => {
      setIsDictating(false);
      setDictateTarget('input');
      alert('Speech recognition error: ' + event.error);
    };
    recognition.onend = () => {
      setIsDictating(false);
      setDictateTarget('input');
    };
    recognitionRef.current = recognition;
    setIsDictating(true);
    setDictateTarget(target === 'input' ? 'input' : messageId);
    recognition.start();
  };

  // Stop dictation
  const handleStopDictation = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsDictating(false);
      setDictateTarget('input');
    }
  };

  // Add a copy button for the last bot response
  const lastBotMessage = messages.length > 0 ? [...messages].reverse().find(m => m.sender === 'bot') : null;

  const [webSearchQuery, setWebSearchQuery] = useState("");
  const [isWebSearching, setIsWebSearching] = useState(false);
  const [inputMode, setInputMode] = useState('chat'); // 'chat' or 'websearch'
  const [showInputMenu, setShowInputMenu] = useState(false);

  // Add state for file upload and chat-with-file
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileUploadLoading, setFileUploadLoading] = useState(false);

  // Handle file upload
  const handleUploadFile = () => {
    if (!isLoggedIn) return;
    fileInputRef.current?.click();
  };

  const onFileInputChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setFileUploadLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    const conversation_id = getValidConversationId(selectedConversation, setSelectedConversation);
    formData.append('conversation_id', conversation_id);
    try {
      const res = await fetch(`${process.env.REACT_APP_BACKEND_URL}/upload_file`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        credentials: 'include',
        body: formData
      });
      const data = await res.json();
      if (res.ok && data.file_id) {
        setUploadedFiles(prev => [...prev, data]);
        setSelectedFile(data);
        // Optionally, show a message in chat about the upload
        setMessages(prev => [...prev, {
          id: Date.now(),
          text: `📁 Uploaded file: ${data.filename}`,
          sender: 'user',
          timestamp: new Date().toLocaleTimeString(),
          hasCode: false,
          reactions: [],
          file_id: data.file_id
        }]);
      } else {
        setMessages(prev => [...prev, {
          id: Date.now(),
          text: `🚨 File upload failed: ${data.error || 'Unknown error'}`,
          sender: 'bot',
          timestamp: new Date().toLocaleTimeString(),
          hasCode: false,
          reactions: [],
        }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, {
        id: Date.now(),
        text: '🚨 File upload failed: Network error',
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        hasCode: false,
        reactions: [],
      }]);
    }
    setFileUploadLoading(false);
  };

  // Handle asking a question about a file
  const handleSendFileQuestion = async () => {
    if (!inputMessage.trim() || isSending || !isLoggedIn || !selectedFile) return;
    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString(),
      hasCode: detectCodeBlocks(inputMessage),
      reactions: [],
      file_id: selectedFile.file_id
    };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsSending(true);
    setIsTyping(true);
    try {
      const conversation_id = getValidConversationId(selectedConversation, setSelectedConversation);
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/chat_with_file`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        credentials: 'include',
        body: JSON.stringify({ question: userMessage.text, file_id: selectedFile.file_id, conversation_id }),
      });
      const data = await response.json();
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: data.answer || (data.error ? `🚨 ${data.error}` : 'No answer.'),
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        hasCode: detectCodeBlocks(data.answer || ''),
        reactions: [],
        file_id: selectedFile.file_id
      }]);
      setIsTyping(false);
      setIsSending(false);
      // If this was a new conversation, update conversation list and select it
      if (!selectedConversation && data.conversation_id) {
        setSelectedConversation(data.conversation_id);
        fetch(`${process.env.REACT_APP_BACKEND_URL}/conversations`, {
          headers: { 'Authorization': `Bearer ${token}` },
          credentials: 'include',
        })
          .then(res => res.json())
          .then(data => setConversations(data.conversations || []));
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: '🚨 File Q&A failed: Network error',
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        isError: true,
        hasCode: false,
        reactions: [],
        file_id: selectedFile.file_id
      }]);
      setIsTyping(false);
      setIsSending(false);
    }
  };

  // Update handleSend to support file Q&A mode
  const handleSend = async () => {
    if (selectedFile) {
      await handleSendFileQuestion();
      return;
    }
    if (inputMode === 'websearch') {
      if (!inputMessage.trim()) return;
      setIsWebSearching(true);
      try {
        const conversation_id = getValidConversationId(selectedConversation, setSelectedConversation);
        // Always send conversation_id if available
        const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/web_search_summarized`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          credentials: 'include',
          body: JSON.stringify({ 
            query: inputMessage, 
            conversation_id,
            gemini_api_key: geminiApiKey || undefined 
          }),
        });
        const data = await response.json();
        // If this was a new conversation, update conversation list and select it
        if (!selectedConversation && data.conversation_id) {
          setSelectedConversation(data.conversation_id);
          // Refetch conversations
          fetch(`${process.env.REACT_APP_BACKEND_URL}/conversations`, {
            headers: { 'Authorization': `Bearer ${token}` },
            credentials: 'include',
          })
            .then(res => res.json())
            .then(data => {
              if (Array.isArray(data.conversations)) {
                setConversations(data.conversations);
              }
            });
        }
        // Refetch chat history for the conversation to include the new messages
        if (data.conversation_id) {
          fetch(`${process.env.REACT_APP_BACKEND_URL}/chat_history?conversation_id=${data.conversation_id}`, {
            headers: { 'Authorization': `Bearer ${token}` },
            credentials: 'include',
          })
            .then(res => res.json())
            .then(data => {
              if (Array.isArray(data.history)) {
                setMessages(data.history);
              }
            });
        }
        setInputMessage("");
      } catch (e) {
        setMessages(prev => [...prev, {
          id: Date.now() + 2,
          text: 'Web search failed. Please try again later.',
          sender: 'bot',
          timestamp: new Date().toLocaleTimeString(),
          hasCode: false,
          reactions: [],
        }]);
      }
      setIsWebSearching(false);
      setInputMode('chat');
      return;
    } else {
      handleSendMessage();
    }
  };

  // In the input area, update the + menu Web Search button to set inputMode to 'websearch' and focus the input
  // Remove the separate web search input rendering
  const navigate = useNavigate ? useNavigate() : null;
  return (
    <div className="app">
      {/* Gemini API Key Input Button */}
      {isLoggedIn && (
        <button
          style={{
            position: 'fixed',
            left: 24,
            bottom: 24,
            zIndex: 1100,
            background: '#232b3a',
            color: ' #667eea',
            border: 'none',
            borderRadius: 12,
            padding: '10px 18px',
            fontWeight: 600,
            fontSize: 15,
            boxShadow: '0 2px 8px rgba(0,0,0,0.18)',
            cursor: 'pointer',
            transition: 'background 0.18s',
          }}
          onClick={() => setShowGeminiKeyInput(v => !v)}
          title="Set Gemini API Key"
        >
          {geminiApiKey ? '🔑 Gemini Key Set' : '🔑 Set Gemini API Key'}
        </button>
      )}
      {/* Gemini API Key Modal */}
      {showGeminiKeyInput && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          background: 'rgba(10,10,20,0.75)',
          zIndex: 1200,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <div style={{
            background: '#181f2a',
            borderRadius: 18,
            boxShadow: '0 8px 32px 0 rgba(102,126,234,0.15)',
            padding: '36px 32px 28px 32px',
            minWidth: 320,
            maxWidth: 370,
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            gap: 18,
            border: '1.5px solid #667eea',
            position: 'relative',
          }}>
            <button
              style={{
                position: 'absolute',
                top: 12,
                right: 12,
                background: 'none',
                border: 'none',
                color: '#fff',
                fontSize: 22,
                cursor: 'pointer',
                opacity: 0.7,
              }}
              onClick={() => setShowGeminiKeyInput(false)}
              title="Close"
            >✕</button>
            <div style={{ fontWeight: 700, fontSize: 18, color: '#4fd1c5', marginBottom: 8 }}>Set Gemini API Key</div>
            <div style={{ color: '#b3b3d1', fontSize: 14, marginBottom: 8 }}>
              Enter your <b>Google Gemini API Key</b> below. This key is used only for your current session and never stored on the server.
            </div>
            <input
              type="password"
              value={geminiApiKey}
              onChange={e => setGeminiApiKey(e.target.value)}
              placeholder="Paste your Gemini API Key here"
              style={{
                padding: '10px 12px',
                borderRadius: 8,
                border: '1.5px solid #2d2d4d',
                background: '#23234a',
                color: '#fff',
                fontSize: 16,
                outline: 'none',
                marginBottom: 12,
              }}
              autoFocus
            />
            <button
              onClick={handleGeminiKeySave}
              style={{
                background: 'linear-gradient(90deg, #667eea 60%, #4fd1c5 100%)',
                color: '#fff',
                fontWeight: 700,
                border: 'none',
                borderRadius: 8,
                padding: '12px 0',
                fontSize: 16,
                cursor: 'pointer',
                boxShadow: '0 2px 8px 0 rgba(102,126,234,0.10)',
                transition: 'background 0.2s, box-shadow 0.2s',
              }}
            >Save Key</button>
          </div>
        </div>
      )}

      {/* Chat History Button - fixed left corner */}
      {isLoggedIn && (
        <button
          className="chat-history-toggle-btn"
          style={{
            position: 'fixed',
            left: 16,
            top: 24,
            zIndex: 1001,
            background: '#181f2a',
            color: '#fff',
            border: 'none',
            borderRadius: '50%',
            width: 48,
            height: 48,
            boxShadow: '0 2px 8px rgba(0,0,0,0.18)',
            fontSize: 24,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onClick={() => setShowSidebar((v) => !v)}
          title={showSidebar ? 'Hide chat history' : 'Show chat history'}
        >
          💬
        </button>
      )}

      {/* Sidebar as overlay, not inline */}
      {isLoggedIn && showSidebar && (
        <div className="sidebar-conversations sidebar-overlay" style={{
          position: 'fixed',
          left: 0,
          top: 0,
          height: '100vh',
          zIndex: 2000,
          boxShadow: '2px 0 16px rgba(0,0,0,0.18)',
          background: 'rgba(24,31,42,0.99)',
          transition: 'transform 0.2s',
          width: 300,
          maxWidth: '90vw',
          display: 'flex',
          flexDirection: 'column',
        }}>
          <div className="sidebar-header" style={{
            padding: '18px 18px 8px 18px',
            fontWeight: 700,
            fontSize: 18,
            color: '#fff',
            borderBottom: '1px solid #232b3a',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <span>Conversations</span>
            <button className="new-conv-btn" onClick={() => { setSelectedConversation(null); setMessages([]); setInputMessage(''); setShowSidebar(false); }} title="Start new conversation" style={{
              background: '#232b3a',
              color: '#fff',
              border: 'none',
              borderRadius: 8,
              fontSize: 22,
              width: 36,
              height: 36,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>＋</button>
          </div>
          <div className="conversation-list" style={{
            flex: 1,
            overflowY: 'auto',
            padding: '8px 0',
          }}>
            {conversations.length === 0 && <div className="no-conv" style={{ color: '#aaa', textAlign: 'center', marginTop: 32 }}>No conversations yet</div>}
            {conversations.map(conv => (
              <div
                key={conv.conversation_id}
                className={`conversation-item${selectedConversation === conv.conversation_id ? ' selected' : ''}`}
                style={{
                  padding: '12px 18px',
                  background: selectedConversation === conv.conversation_id ? '#232b3a' : 'transparent',
                  color: '#fff',
                  cursor: 'pointer',
                  borderLeft: selectedConversation === conv.conversation_id ? '4px solid #4fd1c5' : '4px solid transparent',
                  marginBottom: 2,
                  borderRadius: 6,
                  transition: 'background 0.15s',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 8,
                }}
                onClick={e => {
                  // Prevent click if delete button is pressed
                  if (e.target.classList.contains('delete-conv-btn')) return;
                  setSelectedConversation(conv.conversation_id); setShowSidebar(false);
                }}
              >
                <div style={{flex: 1, minWidth: 0}}>
                  <div className="conv-title" style={{ fontWeight: 600, fontSize: 15, marginBottom: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{conv.last_message?.slice(0, 32) || 'Conversation'}</div>
                  <div className="conv-meta" style={{ fontSize: 12, color: '#8fa2c1', display: 'flex', justifyContent: 'space-between' }}>
                    <span>{new Date(conv.last_time).toLocaleString()}</span>
                    <span>{conv.message_count} msg</span>
                  </div>
                </div>
                <button
                  className="delete-conv-btn"
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#ff5c5c',
                    fontSize: 18,
                    cursor: 'pointer',
                    marginLeft: 8,
                    borderRadius: 6,
                    padding: '2px 6px',
                    transition: 'background 0.15s',
                  }}
                  title="Delete conversation"
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (!window.confirm('Are you sure you want to delete this conversation? This cannot be undone.')) return;
                    try {
                      const res = await fetch(`${process.env.REACT_APP_BACKEND_URL}/conversation/${conv.conversation_id}`, {
                        method: 'DELETE',
                        headers: { 'Authorization': `Bearer ${token}` },
                        credentials: 'include',
                      });
                      if (res.ok) {
                        setConversations(prev => prev.filter(c => c.conversation_id !== conv.conversation_id));
                        if (selectedConversation === conv.conversation_id) {
                          setSelectedConversation(null);
                          setMessages([]);
                        }
                      } else {
                        alert('Failed to delete conversation.');
                      }
                    } catch {
                      alert('Network error while deleting conversation.');
                    }
                  }}
                >🗑️</button>
              </div>
            ))}
          </div>
          <button className="close-sidebar-btn" style={{
            position: 'absolute',
            top: 12,
            right: 12,
            background: 'none',
            border: 'none',
            color: '#fff',
            fontSize: 22,
            cursor: 'pointer',
            opacity: 0.7,
          }} onClick={() => setShowSidebar(false)} title="Close chat history">✕</button>
        </div>
      )}

      {/* User Profile Card - top right */}
      {isLoggedIn && (
        <>
          {/* Avatar button (always visible) */}
          <button
            style={{
              position: 'fixed',
              top: 24,
              right: 24,
              zIndex: 1100,
              width: 48,
              height: 48,
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #232b3a 60%, #4fd1c5 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 26,
              fontWeight: 700,
              color: '#fff',
              boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
              border: '1.5px solid #4fd1c5',
              cursor: 'pointer',
            }}
            onClick={() => setShowProfileCard(true)}
            title="Show profile"
          >
            {username?.[0]?.toUpperCase() || 'U'}
          </button>

          {/* Profile card popup */}
          {showProfileCard && (
            <div
              style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100vw',
                height: '100vh',
                zIndex: 1200,
                background: 'rgba(10,10,20,0.10)',
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'flex-end',
              }}
              onClick={e => {
                // Only close if background is clicked, not the card itself
                if (e.target === e.currentTarget) setShowProfileCard(false);
              }}
            >
              <div
                style={{
                  marginTop: 24,
                  marginRight: 24,
                  background: 'rgba(24,31,42,0.98)',
                  border: '1.5px solid #4fd1c5',
                  borderRadius: 16,
                  boxShadow: '0 4px 16px 0 rgba(79,209,197,0.10)',
                  padding: '18px 24px 16px 24px',
                  minWidth: 220,
                  maxWidth: 320,
                  color: '#fff',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                  gap: 10,
                  position: 'relative',
                }}
                onClick={e => e.stopPropagation()}
              >
                <button
                  style={{
                    position: 'absolute',
                    top: 10,
                    right: 10,
                    background: 'none',
                    border: 'none',
                    color: '#fff',
                    fontSize: 20,
                    cursor: 'pointer',
                    opacity: 0.7,
                  }}
                  onClick={() => setShowProfileCard(false)}
                  title="Close profile"
                >✕</button>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{
                    width: 44,
                    height: 44,
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #232b3a 60%, #4fd1c5 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 26,
                    fontWeight: 700,
                    color: '#fff',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.12)'
                  }}>
                    {username?.[0]?.toUpperCase() || 'U'}
                  </div>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 17, color: '#4fd1c5', marginBottom: 2 }}>{username || 'User'}</div>
                    <div style={{ fontSize: 13, color: '#b3b3d1' }}>CyberGuard Account</div>
                  </div>
                </div>
                <div style={{ width: '100%', marginTop: 8, display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    onClick={() => {
                      setIsLoggedIn(false);
                      setToken('');
                      setMessages([]);
                      setConversations([]);
                      setSelectedConversation(null);
                      setInputMessage('');
                      setUsername("");
                      localStorage.removeItem('geminiApiKey');
                      localStorage.removeItem("username");
                      localStorage.removeItem("token");
                      setShowProfileCard(false);
                    }}
                    style={{
                      background: 'linear-gradient(90deg, #ff5c5c 60%, #4fd1c5 100%)',
                      color: '#fff',
                      fontWeight: 700,
                      border: 'none',
                      borderRadius: 8,
                      padding: '8px 18px',
                      fontSize: 15,
                      cursor: 'pointer',
                      boxShadow: '0 2px 8px 0 rgba(255,92,92,0.10)',
                      transition: 'background 0.2s, box-shadow 0.2s',
                    }}
                  >Logout</button>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Main chat area always present */}
      <div className="main-chat-area" style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        height: '95vh',
        background: 'rgba(18,22,32,0.98)',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* Enhanced Header */}
        <div className="chat-header">
          <div className="header-content">
            <div className="bot-avatar">
              <div className="avatar-ring"></div>
              <div className="avatar-core">🛡️</div>
              <div className="security-pulse"></div>
            </div>
            <div className="header-info">
              <h1 className="bot-name">CyberGuard AI</h1>
              <p className="bot-status">
                <span className="status-dot"></span>
                Secured • Analyzing threats
                <span className="threat-level">
                  <span className="level-indicator green"></span>
                  THREAT LEVEL: LOW
                </span>
              </p>
            </div>
            <div className="security-stats">
              <div className="stat">
                <div className="stat-value">99.9%</div>
                <div className="stat-label">Uptime</div>
              </div>
              <div className="stat">
                <div className="stat-value">{messages.length}</div>
                <div className="stat-label">Messages</div>
              </div>
            </div>
          </div>
        </div>

        {/* Login/Register Form */}
        {!isLoggedIn ? (
          <div className="login-form-container enhanced-form-bg">
            <form className="login-form enhanced-form" onSubmit={showRegister ? handleRegister : handleLogin}>
              <div className="form-header">
                <div className="form-icon">🛡️</div>
                <h2>{showRegister ? 'Create Your CyberGuard Account' : 'Welcome Back to CyberGuard AI'}</h2>
                <p className="form-subtitle">{showRegister ? 'Sign up to start your secure chat journey.' : 'Login to access your secure chat.'}</p>
              </div>
              <div className="form-group">
                <label htmlFor="username">Username</label>
                <input
                  id="username"
                  type="text"
                  autoComplete="username"
                  placeholder="Enter your username"
                  value={showRegister ? registerForm.username : loginForm.username}
                  onChange={e => showRegister
                    ? setRegisterForm(f => ({ ...f, username: e.target.value }))
                    : setLoginForm(f => ({ ...f, username: e.target.value }))}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="password">Password</label>
                <input
                  id="password"
                  type="password"
                  autoComplete={showRegister ? "new-password" : "current-password"}
                  placeholder="Enter your password"
                  value={showRegister ? registerForm.password : loginForm.password}
                  onChange={e => showRegister
                    ? setRegisterForm(f => ({ ...f, password: e.target.value }))
                    : setLoginForm(f => ({ ...f, password: e.target.value }))}
                  required
                />
              </div>
              <button type="submit" className="form-btn main-btn">{showRegister ? 'Register' : 'Login'}</button>
              {showRegister ? (
                <>
                  {registerError && <div className="form-error">{registerError}</div>}
                  {registerSuccess && <div className="form-success">{registerSuccess}</div>}
                  <div className="toggle-link">
                    Already have an account?{' '}
                    <span onClick={() => { setShowRegister(false); setRegisterError(''); setRegisterSuccess(''); }}>
                      <b>Login here</b>
                    </span>
                  </div>
                </>
              ) : (
                <>
                  {loginError && <div className="form-error">{loginError}</div>}
                  <div className="toggle-link">
                    New user?{' '}
                    <span onClick={() => { setShowRegister(true); setLoginError(''); }}>
                      <b>Register here</b>
                    </span>
                  </div>
                </>
              )}
            </form>
          </div>
        ) : (
          <>
            {/* Messages Area - scrollable */}
            {isLoggedIn && (
              <div className="messages-container" style={{
                flex: 1,
                overflowY: 'auto',
                padding: '0 0 0 0',
                margin: 20,
                minHeight: 0,
                maxHeight: 'calc(100vh - 220px)',
                background: 'transparent',
                borderRadius: 0,
                boxShadow: 'none',
              }}>
                <div className="messages-wrapper" style={{ padding: '32px 0 16px 0', minHeight: '100%' }}>
                  {messages.map((message, index) => (
                    <div
                      key={message.id}
                      className={`message-wrapper ${message.sender}`}
                      style={{ animationDelay: `${index * 0.1}s` }}
                    >
                      <div className={`message ${message.sender} ${message.isError ? 'error' : ''} ${message.hasCode ? 'has-code' : ''}`}>
                        {message.sender === 'bot' && (
                          <div className="bot-message-avatar">
                            🛡️
                            <div className="avatar-glow"></div>
                          </div>
                        )}
                        <div className="message-content">
                          <div className="message-text">
                            {renderMessageContent(message, index)}
                            {message.isStreaming && <span className="cursor">|</span>}
                          </div>
                          <div className="message-footer">
                            <div className="message-time">{message.timestamp}</div>
                            {message.sender === 'bot' && !message.isStreaming && (
                              <div className="message-actions">
                                <button 
                                  className="reaction-btn"
                                  onClick={() => setShowReactions(showReactions === message.id ? null : message.id)}
                                >
                                  😊
                                </button>
                                {showReactions === message.id && (
                                  <div className="reactions-popup">
                                    {['👍', '🎯', '🔥', '💡', '❤️'].map(emoji => (
                                      <button
                                        key={emoji}
                                        className="reaction-option"
                                        onClick={() => handleReaction(message.id, emoji)}
                                      >
                                        {emoji}
                                      </button>
                                    ))}
                                  </div>
                                )}
                                <div className="message-reactions">
                                  {message.reactions.map(reaction => (
                                    <span key={reaction.emoji} className="reaction">
                                      {reaction.emoji} {reaction.count}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                  {isTyping && (
                    <div className="message-wrapper bot typing-indicator">
                      <div className="message bot">
                        <div className="bot-message-avatar">
                          🛡️
                          <div className="avatar-glow"></div>
                        </div>
                        <div className="message-content">
                          <div className="typing-animation">
                            <span></span>
                            <span></span>
                            <span></span>
                          </div>
                          <div className="typing-text">Analyzing security patterns...</div>
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              </div>
            )}

            {/* Quick Prompts */}
            <div className="quick-prompts">
              <div className="prompts-label">Quick Security Checks:</div>
              <div className="prompts-container">
                {quickPrompts.map((prompt, index) => (
                  <button
                    key={index}
                    className="quick-prompt"
                    onClick={() => handleQuickPrompt(prompt)}
                    disabled={isSending}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>

            {/* Enhanced Input Area */}
            <div className="input-container">
              <div className="input-wrapper" style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                <textarea
                  ref={textareaRef}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder={inputMode === 'websearch' ? 'Search the web...' : selectedFile ? `Ask about ${selectedFile.filename}...` : 'Describe your security concern, paste code for review, or ask about vulnerabilities...'}
                  className="message-input"
                  rows="1"
                  style={{ height: "24px", minHeight: "20px", maxHeight: "120px" }}
                  disabled={isSending || isWebSearching}
                  autoFocus
                />
                {/* Dictate button */}
                <button
                  onClick={isDictating ? handleStopDictation : () => handleStartDictation('input')}
                  className={`dictate-btn${isDictating ? ' active' : ''}`}
                  style={{
                    marginLeft: 8,
                    background: isDictating ? '#4fd1c5' : '#232b3a',
                    color: isDictating ? '#181f2a' : '#4fd1c5',
                    border: 'none',
                    borderRadius: 8,
                    padding: '8px 12px',
                    fontWeight: 700,
                    fontSize: 16,
                    cursor: 'pointer',
                    boxShadow: isDictating ? '0 2px 8px 0 rgba(79,209,197,0.18)' : 'none',
                    transition: 'background 0.2s, color 0.2s',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                  title={isDictating ? 'Stop dictation' : 'Dictate your message'}
                  disabled={isSending || isWebSearching}
                >
                  {isDictating ? '🛑' : <span style={{fontSize: 22, lineHeight: 1}}>🎙️</span>}
                </button>
                {/* + icon for menu */}
                <div style={{ position: 'relative', marginLeft: 8 }}>
                  <button
                    onClick={() => setShowInputMenu(v => !v)}
                    style={{
                      background: '#232b3a',
                      color: '#4fd1c5',
                      border: 'none',
                      borderRadius: '50%',
                      width: 36,
                      height: 36,
                      fontSize: 24,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                    title="More options"
                  >＋</button>
                  {showInputMenu && (
                    <div ref={inputMenuRef} style={{
                      position: 'absolute',
                      right: 0,
                      bottom: 44, // changed from top: 44 to bottom: 44
                      background: '#181f2a',
                      border: '1.5px solid #4fd1c5',
                      borderRadius: 10,
                      boxShadow: '0 4px 16px 0 rgba(79,209,197,0.10)',
                      padding: '10px 0',
                      minWidth: 180,
                      zIndex: 1000,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 0,
                      maxHeight: 220,
                      overflowY: 'auto',
                    }}>
                      <button
                        onClick={() => {
                          setShowInputMenu(false);
                          setInputMode('websearch');
                          setTimeout(() => textareaRef.current?.focus(), 100);
                        }}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: '#4fd1c5',
                          fontWeight: 600,
                          fontSize: 15,
                          padding: '10px 18px',
                          textAlign: 'left',
                          cursor: 'pointer',
                          width: '100%',
                          borderBottom: '1px solid #232b3a',
                        }}
                      >🌐 Web Search</button>
                      <button
                        onClick={() => { setShowInputMenu(false); handleUploadFile(); }}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: '#4fd1c5',
                          fontWeight: 600,
                          fontSize: 15,
                          padding: '10px 18px',
                          textAlign: 'left',
                          cursor: 'pointer',
                          width: '100%',
                        }}
                      >📁 Upload File</button>
                      <button
                        onClick={() => {
                          setShowInputMenu(false);
                          if (navigate) navigate('/scan-report-review');
                          else window.location.href = '/scan-report-review';
                        }}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: '#4fd1c5',
                          fontWeight: 600,
                          fontSize: 15,
                          padding: '10px 18px',
                          textAlign: 'left',
                          cursor: 'pointer',
                          width: '100%'
                        }}
                      >📝 Scan Report Review</button>
                      {/* Add more tool buttons here as needed */}
                    </div>
                  )}
                </div>
                {/* Send button */}
                <button
                  onClick={handleSend}
                  disabled={(!inputMessage.trim() && inputMode === 'chat') || isSending || isWebSearching}
                  className={`send-button ${isSending || isWebSearching ? 'sending' : ''}`}
                >
                  {(isSending || isWebSearching) ? (
                    <div className="loading-spinner"></div>
                  ) : (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <line x1="22" y1="2" x2="11" y2="13"></line>
                      <polygon points="22,2 15,22 11,13 2,9 22,2"></polygon>
                    </svg>
                  )}
                </button>
                {/* Hidden file input (still needed for upload) */}
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  onChange={onFileInputChange}
                  accept=".pdf,.docx,.txt,.py,.js,.java,.c,.cpp,.go,.rb,.sh,.md,.json,.yaml,.yml"
                />
              </div>
              <div className="input-footer">
                <div className="security-indicator-small">
                  🔒 End-to-end encrypted • 🛡️ Threat monitoring active
                </div>
              </div>
            </div>

            {/* Uploaded files list and selected file context (as before) */}
            {uploadedFiles.length > 0 && (
              <div className="uploaded-files-list" style={{ margin: '8px 0 0 0', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {uploadedFiles.map(f => (
                  <div key={f.file_id} style={{ display: 'inline-flex', alignItems: 'center', position: 'relative' }}>
                    <button
                      onClick={() => setSelectedFile(f)}
                      style={{
                        background: selectedFile && selectedFile.file_id === f.file_id ? '#4fd1c5' : '#232b3a',
                        color: selectedFile && selectedFile.file_id === f.file_id ? '#181f2a' : '#4fd1c5',
                        border: 'none',
                        borderRadius: 8,
                        padding: '6px 14px',
                        fontWeight: 600,
                        fontSize: 14,
                        cursor: 'pointer',
                        boxShadow: selectedFile && selectedFile.file_id === f.file_id ? '0 2px 8px 0 rgba(79,209,197,0.18)' : 'none',
                        marginRight: 2
                      }}
                      title={`Ask about ${f.filename}`}
                    >{f.filename}</button>
                    <button
                      onClick={() => handleRemoveFile(f.file_id)}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: '#ff5c5c',
                        fontSize: 16,
                        marginLeft: -8,
                        cursor: 'pointer',
                        padding: '0 4px',
                        position: 'absolute',
                        right: 0,
                        top: 0
                      }}
                      title={`Remove ${f.filename}`}
                    >✕</button>
                  </div>
                ))}
                <button
                  onClick={() => { setSelectedFile(null); }}
                  style={{
                    background: '#232b3a', color: '#ff5c5c', border: 'none', borderRadius: 8, padding: '6px 14px', fontWeight: 600, fontSize: 14, cursor: 'pointer', marginLeft: 4
                  }}
                  title="Clear file selection"
                >✕</button>
              </div>
            )}
            {selectedFile && (
              <div style={{ margin: '8px 0', color: '#4fd1c5', fontWeight: 600 }}>
                <span>📁 Chatting about: {selectedFile.filename}</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

// PrivateRoute wrapper for protected routes
const PrivateRoute = ({ isLoggedIn, children }) => {
  return isLoggedIn ? children : <Navigate to="/" replace />;
};

// Wrap App in Router and add route for ScanReportReview
const AppWithRouter = () => {
  // Use state lifting to share auth and conversation context
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [selectedConversation, setSelectedConversation] = useState(() => {
    const stored = localStorage.getItem('selectedConversation');
    if (stored && stored !== 'None') return stored;
    const newId = generateUUID();
    localStorage.setItem('selectedConversation', newId);
    return newId;
  });
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState("");
  // Keep localStorage in sync
  React.useEffect(() => {
    if (!selectedConversation || selectedConversation === 'None') {
      const newId = generateUUID();
      setSelectedConversation(newId);
      localStorage.setItem('selectedConversation', newId);
    } else {
      localStorage.setItem('selectedConversation', selectedConversation);
    }
  }, [selectedConversation]);
  // Restore session from localStorage on initial load
  useEffect(() => {
    const storedToken = localStorage.getItem('token');
    const storedUsername = localStorage.getItem('username');
    // Only restore session if token exists and is not undefined/empty
    if (storedToken && storedToken !== 'undefined' && storedToken !== '') {
      setToken(storedToken);
      setIsLoggedIn(true);
      if (storedUsername) setUsername(storedUsername);
    } else {
      setIsLoggedIn(false);
      setToken('');
      setUsername("");
    }
  }, []);
  return (
    <Router>
      <Routes>
        <Route path="/" element={<App token={token} setToken={setToken} selectedConversation={selectedConversation} setSelectedConversation={setSelectedConversation} isLoggedIn={isLoggedIn} setIsLoggedIn={setIsLoggedIn} username={username} setUsername={setUsername} />} />
        <Route path="/scan-report-review" element={
          <PrivateRoute isLoggedIn={isLoggedIn}>
            <ScanReportReview token={token} selectedConversation={selectedConversation} setSelectedConversation={setSelectedConversation} />
          </PrivateRoute>
        } />
      </Routes>
    </Router>
  );
};

export default AppWithRouter;