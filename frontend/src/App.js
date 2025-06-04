import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Welcome message
    setMessages([
      {
        id: 1,
        text: "Hello! I'm your security assistant. I can help you understand vulnerabilities, review code, and provide security guidance. How can I assist you today?",
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString()
      }
    ]);
  }, []);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isSending) return;

    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
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
        isStreaming: true
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
        
        setMessages(prev => prev.map(msg => 
          msg.id === botMessageId 
            ? { ...msg, text: msg.text + chunk }
            : msg
        ));
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
        text: "Sorry, I encountered an error. Please try again.",
        sender: 'bot',
        timestamp: new Date().toLocaleTimeString(),
        isError: true
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
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputMessage]);

  return (
    <div className="app">
      {/* Animated Background */}
      <div className="background">
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
      </div>

      {/* Main Chat Container */}
      <div className="chat-container">
        {/* Header */}
        <div className="chat-header">
          <div className="header-content">
            <div className="bot-avatar">
              <div className="avatar-ring"></div>
              <div className="avatar-core">🛡️</div>
            </div>
            <div className="header-info">
              <h1 className="bot-name">Security Assistant</h1>
              <p className="bot-status">
                <span className="status-dot"></span>
                Online • Ready to help
              </p>
            </div>
          </div>
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
                <div className={`message ${message.sender} ${message.isError ? 'error' : ''}`}>
                  {message.sender === 'bot' && (
                    <div className="bot-message-avatar">🛡️</div>
                  )}
                  <div className="message-content">
                    <div className="message-text">
                      {message.text}
                      {message.isStreaming && <span className="cursor">|</span>}
                    </div>
                    <div className="message-time">{message.timestamp}</div>
                  </div>
                </div>
              </div>
            ))}
            
            {isTyping && (
              <div className="message-wrapper bot typing-indicator">
                <div className="message bot">
                  <div className="bot-message-avatar">🛡️</div>
                  <div className="message-content">
                    <div className="typing-animation">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="input-container">
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about security vulnerabilities, code review, or get security advice..."
              className="message-input"
              rows="1"
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
        </div>
      </div>
    </div>
  );
};

export default App;