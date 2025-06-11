import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ReactMarkdown from 'react-markdown';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [showReactions, setShowReactions] = useState(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginForm, setLoginForm] = useState({ username: '', password: '' });
  const [loginError, setLoginError] = useState('');
  const [token, setToken] = useState('');
  const [showRegister, setShowRegister] = useState(false);
  const [registerForm, setRegisterForm] = useState({ username: '', password: '' });
  const [registerError, setRegisterError] = useState('');
  const [registerSuccess, setRegisterSuccess] = useState('');
  const [conversations, setConversations] = useState([]);
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);
  const [copiedCodeIndex, setCopiedCodeIndex] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

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
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/stream_chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        credentials: 'include',
        body: JSON.stringify({ message: inputMessage, conversation_id: selectedConversation }),
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
        <ReactMarkdown
          children={text}
          components={{
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

  return (
    <div className="app">
      {/* Enhanced Animated Background */}
      <div className="background">
        <div className="cyber-grid"></div>
        <div className="stars"></div>
        <div className="floating-shapes">
          <div className="shape shape-1"></div>
          <div className="shape shape-2"></div>
          <div className="shape shape-3"></div>
          <div className="shape shape-4"></div>
          <div className="shape shape-5"></div>
          <div className="shape shape-6"></div>
        </div>
        <div className="gradient-orb orb-1"></div>
        <div className="gradient-orb orb-2"></div>
        <div className="gradient-orb orb-3"></div>
        <div className="security-particles">
          <div className="particle"></div>
          <div className="particle"></div>
          <div className="particle"></div>
          <div className="particle"></div>
          <div className="particle"></div>
        </div>
      </div>

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
                }}
                onClick={() => { setSelectedConversation(conv.conversation_id); setShowSidebar(false); }}
              >
                <div className="conv-title" style={{ fontWeight: 600, fontSize: 15, marginBottom: 2 }}>{conv.last_message?.slice(0, 32) || 'Conversation'}</div>
                <div className="conv-meta" style={{ fontSize: 12, color: '#8fa2c1', display: 'flex', justifyContent: 'space-between' }}>
                  <span>{new Date(conv.last_time).toLocaleString()}</span>
                  <span>{conv.message_count} msg</span>
                </div>
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
              <div className="input-wrapper">
                <textarea
                  ref={textareaRef}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Describe your security concern, paste code for review, or ask about vulnerabilities..."
                  className="message-input"
                  rows="1"
                  style={{ height: "35px", minHeight: "24px", maxHeight: "120px" }}
                  disabled={isSending}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim() || isSending}
                  className={`send-button ${isSending ? 'sending' : ''}`}
                >
                  {isSending ? (
                    <div className="loading-spinner"></div>
                  ) : (
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <line x1="22" y1="2" x2="11" y2="13"></line>
                      <polygon points="22,2 15,22 11,13 2,9 22,2"></polygon>
                    </svg>
                  )}
                </button>
              </div>
              <div className="input-footer">
                <div className="security-indicator-small">
                  🔒 End-to-end encrypted • 🛡️ Threat monitoring active
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default App;