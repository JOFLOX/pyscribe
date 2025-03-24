# PyScribe 📝 - AI-Powered Python Documentation Generator

[![Project Status](https://img.shields.io/badge/status-active-%2300cc00)](https://github.com/JOFLOX/PyScribe)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)

Generate professional documentation and READMEs for Python projects using AI 🤖. Supports both file uploads and GitHub repository integration!

## 🚀 Key Features
- **AI-Powered Documentation** - Automatically generate comprehensive docs using LLMs
- **Customizable Templates** - Modify prompts for documentation style and content
- **Markdown Formatting** - Clean, well-structured output with emoji support ✨
- **GitHub Integration** - Clone repos directly for documentation generation
- **Real-time Editing** - Preview and edit generated documentation before export

## 📋 Quick Start

### Installation

```

# Clone repository
git clone https://github.com/JOFLOX/PyScribe.git

# Install dependencies
cd PyScribe
pip install -r requirements.txt

# Set up environment variables
echo "OPENROUTER_API_KEY=your_api_key_here" > .env


```

### Basic Usage

```
streamlit run app.py


```
1. Upload Python files or enter GitHub repo URL
2. Configure AI settings (optional)
3. Click "Generate Documentation"
4. Review/edit generated docs
5. Export Markdown files

## 🔧 Technical Architecture


```
graph TD
    A[Streamlit UI] --> B[Documentation Generator]
    B --> C[OpenRouter API]
    B --> D[Markdown Sanitizer]
    B --> E[GitHub Integration]
    D --> F[Output Formatter]
    F --> G[User Download]


```

### Main Components
| Component | Description |
|-----------|-------------|
| `app.py` | Main Streamlit application logic |
| `generate_documentation()` | Handles AI API interactions |
| `sanitize_markdown()` | Ensures proper Markdown formatting |
| `clone_repository()` | GitHub integration handler |

### Dependencies
| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `requests` | API communication |
| `python-dotenv` | Environment management |
| `GitPython` | Repository handling |
| `regex` | Text processing |

## 💻 Developer Guide

### API Documentation

```
def generate_documentation(code: str) -> str:
    """Generate docs using OpenRouter API
    
    Args:
        code: Python source code to document
        
    Returns:
        Formatted Markdown documentation
    """


```

### Contribution Guidelines
1. Fork the repository
2. Create feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add some feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open Pull Request



```

# Run Streamlit in development mode
STREAMLIT_DEBUG=1 streamlit run app.py

# Run unit tests (coming soon)

# pytest tests/


```

---

[📘 Full Documentation](docs/GUIDE.md) | [🐛 Report Issues](https://github.com/JOFLOX/PyScribe/issues) | [💡 Feature Requests](https://github.com/JOFLOX/PyScribe/discussions)

_Made with ❤️ by [Youssef Sayed Gharib](https://www.linkedin.com/in/youssef-sayed-joe)_
```
