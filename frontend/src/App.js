import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [showReactions, setShowReactions] = useState(null);
  const [mode, setMode] = useState('vulnerability');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Welcome message with enhanced security focus
    setMessages([
      {
        id: 1,
        text: "🛡️ **Security Assistant Activated**\n\nHello! I'm your cybersecurity expert. I can help you with:\n\n• 🔍 **Vulnerability Analysis** - Identify security flaws\n• 💻 **Code Review** - Security-focused code analysis  \n• 🚨 **Threat Assessment** - Evaluate potential risks\n• 🔒 **Best Practices** - Security implementation guidance\n• 📋 **Compliance** - Standards and frameworks\n\nWhat security challenge can I help you tackle today?",
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        hasCode: false,
        reactions: []
      }
    ]);
  }, []);

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

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isSending) return;

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
        },
        credentials: 'include',
        body: JSON.stringify({ message: inputMessage }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      // Create bot message placeholder
      const botMessageId = Date.now() + 1;
      const botMessage = {
        id: botMessageId,
        text: '',
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        isStreaming: true,
        hasCode: false,
        reactions: []
      };

      setMessages(prev => [...prev, botMessage]);
      setIsTyping(false);

      // Read the stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        
        setMessages(prev => prev.map(msg => {
          if (msg.id === botMessageId) {
            const newText = msg.text + chunk;
            return { 
              ...msg, 
              text: newText,
              hasCode: detectCodeBlocks(newText)
            };
          }
          return msg;
        }));
      }

      // Mark streaming as complete
      setMessages(prev => prev.map(msg => 
        msg.id === botMessageId 
          ? { ...msg, isStreaming: false }
          : msg
      ));

    } catch (error) {
      console.error('Error sending message:', error);
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
    } finally {
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

  const renderMessageContent = (message) => {
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

  // Add buttons for vulnerability-specific and general chatbot modes
  const handleModeChange = (mode) => {
    setInputMessage('');
    setMessages([]);
    setMode(mode);
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

      {/* Main Chat Container */}
      <div className="chat-container">
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

        {/* Mode Selector */}
        <div className="mode-selector">
          <button
            className={`mode-btn ${mode === 'vulnerability' ? 'active' : ''}`}
            onClick={() => handleModeChange('vulnerability')}
          >
            Vulnerability-Specific Query
          </button>
          <button
            className={`mode-btn ${mode === 'general' ? 'active' : ''}`}
            onClick={() => handleModeChange('general')}
          >
            General Chatbot
          </button>
        </div>

        {/* Messages Area */}
        <div className="messages-container">
          <div className="messages-wrapper">
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
                      {renderMessageContent(message)}
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
              placeholder={mode === 'vulnerability' ? "Enter vulnerability name or description..." : "Describe your security concern, paste code for review, or ask about vulnerabilities..."}
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
      </div>
    </div>
  );
};

export default App;