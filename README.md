# 🛡️ CyberGuard AI - Advanced Cybersecurity Assistant

CyberGuard AI is a comprehensive cybersecurity solution that provides automated vulnerability scanning, analysis, and remediation for software applications. This fullstack application integrates AI-powered security assessment tools to identify potential threats, analyze code vulnerabilities, and recommend fixes with detailed explanations.

## ✨ Features

### 🔐 Authentication & Session Management
- Secure JWT-based authentication
- Persistent login sessions with localStorage
- Protected routes and secure session handling
- User registration and profile management

### 💬 AI-Powered Chat Interface
- Real-time chat with CyberGuard AI
- Context-aware security advice
- Vulnerability-specific guidance
- Code analysis and recommendations
- Conversation history and management

### 📄 Scan Report Analysis
- Upload and analyze Checkmarx scan reports (JSON/PDF)
- Extract code snippets and vulnerabilities
- Side-by-side PDF preview and findings review
- AI-powered explanations and patch recommendations
- Three-column dashboard layout for comprehensive analysis

### 🔍 Advanced Security Features
- OWASP Top 10 vulnerability detection
- CVSS scoring and risk assessment
- Real-time threat analysis with Gemini AI integration
- Code snippet extraction and analysis
- Remediation strategies and best practices

### 🎨 Modern UI/UX
- Dark theme cybersecurity interface
- Responsive design for all devices
- Markdown rendering for rich content
- Syntax highlighting for code blocks
- Interactive security dashboard

## 🏗️ Architecture

### Backend (Python/Flask)
- **Framework**: Flask with MongoDB
- **AI Models**: Ollama (Llama 3) + Google Gemini API
- **Authentication**: JWT tokens with bcrypt password hashing
- **Database**: MongoDB with schema validation
- **Features**: File upload, PDF parsing, vulnerability analysis

### Frontend (React)
- **Framework**: React 19 with React Router
- **Styling**: Custom CSS with modern design
- **Components**: Modular architecture with reusable components
- **State Management**: React hooks and localStorage
- **PDF Handling**: react-pdf and pdfjs-dist

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB
- Ollama with Llama 3 model

### Automated Setup (Windows)
```powershell
# Clone the repository
git clone <repository-url>
cd EMERGENT

# Run the automated setup script
.\start.ps1
```

### Manual Setup

#### 1. Backend Setup
```bash
cd backend
pip install -r requirements_clean.txt

# Start the backend server
python app1.py
```

#### 2. Frontend Setup
```bash
cd frontend
npm install

# Start the frontend server
npm start
```

#### 3. Database Setup
Ensure MongoDB is running on port 27017. The application will automatically create required collections with schema validation.

#### 4. AI Model Setup
```bash
# Install and pull Ollama model
ollama pull llama3
```

## 🔧 Configuration

### Environment Variables

#### Backend (.env)
```env
MONGO_URI=mongodb://localhost:27017
JWT_SECRET=your-jwt-secret-key
GEMINI_API_KEY=your-gemini-api-key  # Optional
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

#### Frontend (.env)
```env
REACT_APP_BACKEND_URL=http://localhost:9000
WDS_SOCKET_PORT=443
```

## 📚 API Endpoints

### Authentication
- `POST /register` - User registration
- `POST /login` - User login
- `GET /conversations` - List user conversations
- `DELETE /conversation/<id>` - Delete conversation

### Chat & AI
- `POST /stream_chat` - Send message to AI
- `GET /chat_history` - Get conversation history
- `POST /web_search_summarized` - Web search with AI summary

### Scan Reports
- `POST /upload_scan_report` - Upload scan report
- `GET /scan_findings` - Get vulnerability findings
- `POST /scan_finding/<id>/explain` - Get vulnerability explanation
- `POST /scan_finding/<id>/patch` - Get patch recommendation

### System
- `GET /api/health` - Health check
- `GET /api/vulnerabilities` - List known vulnerabilities
- `GET /` - Service information

## 🗂️ Project Structure

```
EMERGENT/
├── backend/
│   ├── app1.py                 # Main Flask application
│   ├── requirements_clean.txt  # Python dependencies
│   └── .env                    # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── ScanReportReview.js # Scan report interface
│   │   ├── MarkdownRenderer.js # Markdown component
│   │   └── App.css            # Styling
│   ├── public/
│   │   ├── index.html         # HTML template
│   │   └── Logo.png           # Application logo
│   ├── package.json           # NPM dependencies
│   └── .env                   # Environment variables
├── scripts/
├── tests/
├── start.ps1                  # Automated startup script
└── README.md                  # Project documentation
```

## 🛡️ Security Features

### Vulnerability Database
- SQL Injection detection and prevention
- Cross-Site Scripting (XSS) protection
- Cross-Site Request Forgery (CSRF) mitigation
- Buffer Overflow analysis
- Command Injection detection

### Risk Assessment
- CVSS v3.1 scoring
- Severity classification (Critical, High, Medium, Low)
- Real-time threat analysis
- Security best practices recommendations

## 🔌 Integrations

### AI Services
- **Ollama**: Local LLM for privacy-focused AI responses
- **Google Gemini**: Enhanced context retrieval and analysis
- **Sentence Transformers**: Semantic similarity for caching

### Security Tools
- **Checkmarx**: Scan report analysis and parsing
- **OWASP**: Security framework integration
- **CVE Database**: Vulnerability reference

## 🚨 Troubleshooting

### Common Issues

#### 401 Unauthorized Errors
- Check JWT token in localStorage
- Verify backend JWT_SECRET configuration
- Ensure user is logged in

#### MongoDB Connection Issues
- Verify MongoDB is running on port 27017
- Check MONGO_URI environment variable
- Ensure MongoDB collections are created

#### Ollama Model Issues
- Install Ollama: `ollama pull llama3`
- Check Ollama service is running
- Verify OLLAMA_BASE_URL configuration

#### Frontend Build Issues
- Clear node_modules: `rm -rf node_modules && npm install`
- Check React version compatibility
- Verify all dependencies are installed

## 📊 Performance Optimization

### Backend
- MongoDB connection pooling
- JWT token caching
- Gemini API response caching
- Efficient PDF parsing with pdfplumber

### Frontend
- Code splitting with React.lazy
- Optimized re-renders with React.memo
- localStorage for session persistence
- Efficient state management

## 🔄 Updates & Maintenance

### Version History
- **v3.0.0**: Full-featured release with scan report analysis
- **v2.1.0**: Enhanced UI and session management
- **v2.0.0**: Gemini AI integration
- **v1.0.0**: Initial release with basic chat functionality

### Future Enhancements
- Multi-language support
- Advanced vulnerability scanning
- Integration with more security tools
- Real-time collaboration features
- Enhanced reporting capabilities

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

## 🙏 Acknowledgments

- OpenAI for AI model inspiration
- OWASP for security frameworks
- MongoDB for database solutions
- React team for the frontend framework
- Flask team for the backend framework

---

**🛡️ Stay Secure with CyberGuard AI!**
