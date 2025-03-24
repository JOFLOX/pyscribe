# Import required libraries
import streamlit as st          # Web app framework
import requests                 # For making HTTP requests to the API
import os                      # For operating system operations
from dotenv import load_dotenv # For loading environment variables
import json                    # For JSON handling
import time                    # For adding delays between API calls
import tempfile               # For creating temporary directories
from git import Repo          # For Git operations
import shutil                 # For file operations
from typing import Dict, Any  # For type hints
import re                     # For regular expression operations

# Load environment variables from .env file
load_dotenv()

# Configure the Streamlit page settings
st.set_page_config(
    page_title="PyScribe - Python Documentation Generator",  # Browser tab title
    page_icon="📝",                                         # Browser tab icon
    layout="wide"                                           # Use wide layout
)

def sanitize_markdown(content: str) -> str:
    """
    Ensures markdown is properly formatted by:
    - Adding proper spacing around headers
    - Managing code blocks
    - Handling lists
    - Removing excessive blank lines
    """
    # Split content into individual lines for processing
    lines = content.split('\n')
    sanitized_lines = []
    open_code_blocks = 0      # Track nested code blocks
    in_list = False          # Track if currently in a list
    prev_line_empty = True   # Track if previous line was empty
    
    # Process each line
    for i, line in enumerate(lines):
        current_line = line
        stripped_line = line.strip()
        
        # Handle code blocks (```)
        if stripped_line.startswith('```'):
            if open_code_blocks == 0:
                # Opening a new code block
                open_code_blocks += 1
                # Add blank line before code block if needed
                if sanitized_lines and not prev_line_empty:
                    sanitized_lines.append('')
            else:
                # Closing a code block
                open_code_blocks -= 1
                # Ensure proper closing syntax
                if not stripped_line.endswith('```'):
                    current_line = '```'
        
        # Handle headers (#, ##, etc.)
        if not open_code_blocks and stripped_line.startswith('#'):
            # Fix header formatting (ensure space after #)
            header_match = re.match(r'^(#+)(.*)$', stripped_line)
            if header_match:
                hashes, content = header_match.groups()
                current_line = f"{hashes} {content.lstrip()}"
            
            # Ensure blank line before header if not at start
            if sanitized_lines and not prev_line_empty:
                sanitized_lines.append('')
        
        # Handle lists
        if not open_code_blocks and (stripped_line.startswith('- ') or 
                                   stripped_line.startswith('* ') or 
                                   re.match(r'^\d+\.', stripped_line)):
            if not in_list and sanitized_lines and not prev_line_empty:
                sanitized_lines.append('')
            in_list = True
        else:
            if in_list and stripped_line:
                in_list = False
                if not stripped_line.startswith(' '):  # Not a list continuation
                    sanitized_lines.append('')
        
        # Add the current line
        sanitized_lines.append(current_line)
        
        # Track if this line was empty for next iteration
        prev_line_empty = not stripped_line
    
    # Close any remaining open code blocks
    if open_code_blocks > 0:
        if not prev_line_empty:
            sanitized_lines.append('')
        sanitized_lines.append('```')
    
    # Join lines back together
    content = '\n'.join(sanitized_lines)
    
    # Remove more than two consecutive blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    # Ensure proper spacing around code blocks
    content = re.sub(r'\n```(\w+)\n', r'\n\n```\1\n', content)
    content = re.sub(r'\n```\n', r'\n\n```\n', content)
    
    # Ensure document ends with newline
    if not content.endswith('\n'):
        content += '\n'
    
    return content

def generate_documentation(code: str) -> str:
    """Generate documentation using OpenRouter API."""
    # Check for custom API key first
    custom_api_key = st.session_state.llm_settings.get('custom_api_key')
    api_key = custom_api_key if custom_api_key else os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("No API key available. Please provide a custom API key or set OPENROUTER_API_KEY in environment variables")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://pyscribe.com",
        "X-Title": "PyScribe Documentation Generator"
    }
    
    # Get the custom documentation prompt from settings and ensure it's the latest version
    documentation_prompt = st.session_state.llm_settings.get('documentation_prompt', '')
    if not documentation_prompt:
        st.warning("Using default documentation prompt as custom prompt is empty")
        documentation_prompt = """Create comprehensive Python documentation using markdown and emojis..."""
    
    # Create the full prompt
    prompt = f"""{documentation_prompt.strip()}

Here's the code to document:

{code}"""
    
    payload = {
        "model": st.session_state.llm_settings['model'],
        "messages": [
            {"role": "system", "content": "You are a documentation expert. Generate clear and comprehensive documentation."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": st.session_state.llm_settings['max_tokens'],
        "temperature": st.session_state.llm_settings['temperature'],
        "presence_penalty": st.session_state.llm_settings['presence_penalty'],
        "frequency_penalty": st.session_state.llm_settings['frequency_penalty']
    }
    
    for attempt in range(3):
        try:
            st.write(f"Attempt {attempt + 1} of 3...")
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120  # Increased timeout
            )
            
            # Log response details
            st.write(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                try:
                    # Handle different possible response structures
                    if "choices" in response_data:
                        if isinstance(response_data["choices"], list) and len(response_data["choices"]) > 0:
                            choice = response_data["choices"][0]
                            if isinstance(choice, dict):
                                # Standard OpenAI-like format
                                if "message" in choice and "content" in choice["message"]:
                                    content = choice["message"]["content"]
                                # Alternative format some models might use
                                elif "text" in choice:
                                    content = choice["text"]
                                else:
                                    raise ValueError("Unexpected choice format in response")
                            else:
                                raise ValueError("Choice is not a dictionary")
                        else:
                            raise ValueError("No choices in response")
                    else:
                        raise ValueError("No choices field in response")
                    
                    if content.strip():
                        return content
                    st.write("Received empty response")
                except Exception as e:
                    st.write(f"Error parsing response: {str(e)}")
                    st.write(f"Response structure: {response_data}")
            elif response.status_code == 429:
                wait_time = 2 ** attempt
                st.write(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            elif response.status_code == 401:
                raise ValueError("Invalid API key. Please check your OPENROUTER_API_KEY")
            else:
                st.write(f"API Error: {response.text}")
                time.sleep(2)
                
        except requests.exceptions.Timeout:
            st.write("Request timed out. Retrying...")
        except requests.exceptions.RequestException as e:
            st.write(f"Network error: {str(e)}")
        except Exception as e:
            st.write(f"Unexpected error: {str(e)}")
        
        time.sleep(2)  # Brief pause between attempts
    
    raise Exception(
        "Failed to generate documentation. Please check your API key and internet connection, "
        "or try again later."
    )

def generate_readme(files_content: dict) -> str:
    """Generate README based on all uploaded files."""
    # Check for custom API key first
    custom_api_key = st.session_state.llm_settings.get('custom_api_key')
    api_key = custom_api_key if custom_api_key else os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("No API key available. Please provide a custom API key or set OPENROUTER_API_KEY in environment variables")
    
    combined_content = "\n\n".join([
        f"File: {filename}\n```python\n{content}\n```" 
        for filename, content in files_content.items()
    ])
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://pyscribe.com",
        "X-Title": "PyScribe Documentation Generator"
    }
    
    # Get the custom README prompt from settings and ensure it's the latest version
    readme_prompt = st.session_state.llm_settings.get('readme_prompt', '')
    if not readme_prompt:
        st.warning("Using default README prompt as custom prompt is empty")
        readme_prompt = """Create a professional GitHub README.md using markdown and emojis..."""
    
    # Create the full prompt
    prompt = f"""{readme_prompt.strip()}

Here are the project files:

{combined_content}"""
    
    payload = {
        "model": st.session_state.llm_settings['model'],
        "messages": [
            {"role": "system", "content": "You are a documentation expert. Generate clear and comprehensive documentation."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": st.session_state.llm_settings['max_tokens'],
        "temperature": st.session_state.llm_settings['temperature'],
        "presence_penalty": st.session_state.llm_settings['presence_penalty'],
        "frequency_penalty": st.session_state.llm_settings['frequency_penalty']
    }
    
    for attempt in range(3):
        try:
            st.write(f"Generating README - Attempt {attempt + 1} of 3...")
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120  # Increased timeout
            )
            
            st.write(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                response_data = response.json()
                try:
                    # Handle different possible response structures
                    if "choices" in response_data:
                        if isinstance(response_data["choices"], list) and len(response_data["choices"]) > 0:
                            choice = response_data["choices"][0]
                            if isinstance(choice, dict):
                                # Standard OpenAI-like format
                                if "message" in choice and "content" in choice["message"]:
                                    content = choice["message"]["content"]
                                # Alternative format some models might use
                                elif "text" in choice:
                                    content = choice["text"]
                                else:
                                    raise ValueError("Unexpected choice format in response")
                            else:
                                raise ValueError("Choice is not a dictionary")
                        else:
                            raise ValueError("No choices in response")
                    else:
                        raise ValueError("No choices field in response")
                    
                    if content.strip():
                        return content
                    st.write("Received empty response")
                except Exception as e:
                    st.write(f"Error parsing response: {str(e)}")
                    st.write(f"Response structure: {response_data}")
            elif response.status_code == 429:
                wait_time = 2 ** attempt
                st.write(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            elif response.status_code == 401:
                raise ValueError("Invalid API key. Please check your OPENROUTER_API_KEY")
            else:
                st.write(f"API Error: {response.text}")
                time.sleep(2)
                
        except requests.exceptions.Timeout:
            st.write("Request timed out. Retrying...")
        except requests.exceptions.RequestException as e:
            st.write(f"Network error: {str(e)}")
        except Exception as e:
            st.write(f"Unexpected error: {str(e)}")
        
        time.sleep(2)
    
    raise Exception(
        "Failed to generate README. Please check your API key and internet connection, "
        "or try again later."
    )

def clone_repository(repo_url: str) -> dict:
    """Clone a GitHub repository and return its Python files content."""
    files_content = {}
    
    # Creates a temporary directory that will be automatically deleted
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Clone the repository into the temporary directory
            st.write(f"Cloning repository to: {temp_dir}")  # Add this for debugging
            repo = Repo.clone_from(repo_url, temp_dir)
            
            # Walk through the repository and find Python files
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        # Get relative path for display
                        rel_path = os.path.relpath(file_path, temp_dir)
                        
                        # Read file content
                        with open(file_path, 'r', encoding='utf-8') as f:
                            files_content[rel_path] = f.read()
            
            return files_content
            
        except Exception as e:
            raise Exception(f"Failed to clone repository: {str(e)}")
        # The temporary directory is automatically deleted when exiting this block

def process_files(files_content: dict):
    """Process the files and generate documentation."""
    try:
        with st.spinner("Generating documentation..."):
            # Generate documentation for each file
            for filename, content in files_content.items():
                doc = generate_documentation(content)
                # Sanitize and format markdown
                doc = sanitize_markdown(doc)
                st.session_state.files_content[filename] = doc
                st.session_state.edited_content[filename] = doc
            
            # Generate README
            readme = generate_readme(files_content)
            # Sanitize and format markdown
            readme = sanitize_markdown(readme)
            st.session_state.readme_content = readme
            st.session_state.generated = True
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return

def initialize_llm_settings():
    """Initialize default LLM settings if not present in session state."""
    DEFAULT_DOCUMENTATION_PROMPT = """Create comprehensive Python documentation using markdown and emojis. Include these sections:

    📚 Module Overview
    - Brief description of the module's purpose
    - Key features and functionality
    - Main components and their relationships

    🔧 Technical Details
    - Classes and methods (with types and parameters)
    - Functions and their usage
    - Dependencies and requirements
    - Error handling patterns

    💡 Usage Examples
    - Code snippets showing common use cases
    - Integration examples
    - Best practices

    ⚙️ Setup & Configuration
    - Installation steps
    - Environment requirements
    - Configuration options

    Use markdown formatting:
    - Code blocks with ```python
    - Headers with proper hierarchy (#, ##, ###)
    - Lists and tables where appropriate
    - Backticks for inline code
    - Leave blank lines around sections"""

    DEFAULT_README_PROMPT = """Create a professional GitHub README.md using markdown and emojis. Include these sections:

    🚀 Project Title
    - Project name with descriptive emoji
    - Brief, compelling project description
    - Key features and benefits

    📋 Quick Start
    - Installation steps
    - Basic usage example
    - Configuration requirements

    🔧 Technical Details
    - System architecture
    - Main components
    - External dependencies

    💻 Developer Guide
    - Setup instructions
    - API documentation
    - Contributing guidelines

    Use markdown formatting:
    - Code blocks with language specification
    - Clear header hierarchy
    - Lists and tables where needed
    - Badges for status/version
    - Links to important sections"""

    if 'llm_settings' not in st.session_state:
        st.session_state.llm_settings = {
            'model': "deepseek/deepseek-r1:free",
            'max_tokens': 4000,
            'temperature': 0.7,
            'presence_penalty': 0.6,
            'frequency_penalty': 0.3,
            'custom_api_key': "",
            'documentation_prompt': DEFAULT_DOCUMENTATION_PROMPT,
            'readme_prompt': DEFAULT_README_PROMPT
        }
    else:
        # Only reset the prompts while keeping other settings intact
        st.session_state.llm_settings['documentation_prompt'] = DEFAULT_DOCUMENTATION_PROMPT
        st.session_state.llm_settings['readme_prompt'] = DEFAULT_README_PROMPT

def show_settings_modal():
    """Display the settings modal with LLM parameters."""
    st.markdown("### ⚙️ LLM Settings")
    
    # Create tabs for different settings categories
    settings_tab, api_tab = st.tabs(["Model Settings", "API Settings"])
    
    with settings_tab:
        # Model selection
        st.session_state.llm_settings['model'] = st.selectbox(
            "Model",
            ["deepseek/deepseek-r1:free", "open-r1/olympiccoder-32b:free", "mistralai/mistral-small-3.1-24b-instruct:free"],
            index=0,
            help="Select the AI model to use for generation"
        )
        
        # Sliders for numerical parameters
        st.session_state.llm_settings['max_tokens'] = st.slider(
            "Max Tokens",
            min_value=1000,
            max_value=8000,
            value=st.session_state.llm_settings['max_tokens'],
            step=100,
            help="Maximum number of tokens in the response"
        )
        
        st.session_state.llm_settings['temperature'] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.llm_settings['temperature'],
            step=0.1,
            help="Controls randomness in the output (0 = deterministic, 1 = creative)"
        )
        
        st.session_state.llm_settings['presence_penalty'] = st.slider(
            "Presence Penalty",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.llm_settings['presence_penalty'],
            step=0.1,
            help="Penalizes new tokens based on whether they appear in the text so far"
        )
        
        st.session_state.llm_settings['frequency_penalty'] = st.slider(
            "Frequency Penalty",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.llm_settings['frequency_penalty'],
            step=0.1,
            help="Penalizes new tokens based on their frequency in the text so far"
        )
    
    with api_tab:
        st.markdown("#### 🔑 API Configuration")
        
        # Add API key input
        custom_api_key = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.llm_settings.get('custom_api_key', ''),
            type="password",
            help="Enter your OpenRouter API key. Leave empty to use the default key."
        )
        
        # Update the API key in session state
        st.session_state.llm_settings['custom_api_key'] = custom_api_key
        
        # Add link to OpenRouter
        st.markdown("""
        Don't have an API key? Get one from [OpenRouter](https://openrouter.ai/models) 🔗
        
        **Note:** If no API key is provided, the default key will be used.
        """)

def preview_markdown(content: str) -> str:
    """Create a preview version of the markdown with visible formatting markers."""
    # Add visual indicators for markdown elements
    preview = content
    preview = re.sub(r'```(\w+)', r'📝 ```\1', preview)  # Mark code blocks
    preview = re.sub(r'```$', r'```  ⏹️', preview, flags=re.MULTILINE)  # Mark code block ends
    preview = re.sub(r'^(#+)\s', r'📌 \1 ', preview, flags=re.MULTILINE)  # Mark headers
    preview = re.sub(r'^\*\s', r'🔸 * ', preview, flags=re.MULTILINE)  # Mark list items
    return preview

def main():
    # Title and description section with centered layout
    st.markdown("""
        <h1 style='text-align: center;'>PyScribe 📝</h1>
        <h3 style='text-align: center;'>AI-Powered Python Documentation Generator</h3>
        <p style='text-align: center;'>By Youssef Sayed Gharib</p>
        <p style='text-align: center;'>
            <a href="https://www.linkedin.com/in/youssef-sayed-joe" target="_blank">👔 LinkedIn</a> • 
            <a href="https://github.com/JOFLOX" target="_blank">👨‍💻 GitHub</a>
        </p>
        <br>
    """, unsafe_allow_html=True)
    
    # Initialize all session states
    initialize_llm_settings()
    if 'files_content' not in st.session_state:
        st.session_state.files_content = {}
    if 'edited_content' not in st.session_state:
        st.session_state.edited_content = {}
    if 'readme_content' not in st.session_state:
        st.session_state.readme_content = ""
    if 'show_preview' not in st.session_state:
        st.session_state.show_preview = {}
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False
    # Add this line to initialize the preview toggle state
    if 'preview_toggle_readme' not in st.session_state:
        st.session_state.preview_toggle_readme = False

    # Settings button in the header
    col1, col2 = st.columns([0.9, 0.1])
    with col2:
        if st.button("⚙️", help="Configure LLM Settings"):
            st.session_state.show_settings = not st.session_state.show_settings
    
    # Show settings modal if enabled
    if st.session_state.show_settings:
        show_settings_modal()
        st.markdown("---")  # Add separator after settings

    # Add tabs for different input methods
    input_method = st.radio(
        "Choose input method:",
        ["Upload Files", "GitHub Repository"],
        help="Select how you want to provide your Python files"
    )
    
    if input_method == "Upload Files":
        # Existing file upload logic
        uploaded_files = st.file_uploader(
            "Upload Python files",
            type=["py"],
            accept_multiple_files=True,
            help="Select one or multiple Python files"
        )
        
        if uploaded_files:
            files_content = {}
            for file in uploaded_files:
                content = file.read().decode("utf-8")
                files_content[file.name] = content
                
            if st.button("Generate Documentation", type="primary"):
                process_files(files_content)
                
    else:  # GitHub Repository
        repo_url = st.text_input(
            "Enter GitHub repository URL",
            help="e.g., https://github.com/username/repository"
        )
        
        if repo_url:
            if st.button("Generate Documentation", type="primary"):
                try:
                    with st.spinner("Cloning repository and analyzing files..."):
                        files_content = clone_repository(repo_url)
                        if files_content:
                            process_files(files_content)
                        else:
                            st.error("No Python files found in the repository")
                except Exception as e:
                    st.error(f"Error processing repository: {str(e)}")

    # Display generated content
    if st.session_state.get('generated', False):
        tab1, tab2 = st.tabs(["Documentation", "README"])
        
        with tab1:
            for filename in list(st.session_state.files_content.keys()):
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.subheader(f"Documentation for {filename}")
                with col2:
                    if st.button("Close", key=f"close_{filename}"):
                        del st.session_state.files_content[filename]
                        del st.session_state.edited_content[filename]
                        if filename in st.session_state.show_preview:
                            del st.session_state.show_preview[filename]
                        st.rerun()
                
                # Edit box first
                edited_doc = st.text_area(
                    "Edit documentation:",
                    value=st.session_state.edited_content[filename],
                    height=300,
                    key=f"edit_{filename}"
                )
                
                # Update edited content in session state
                if edited_doc != st.session_state.edited_content[filename]:
                    st.session_state.edited_content[filename] = edited_doc
                
                # Preview/Apply button
                if st.button("Apply Changes", key=f"preview_{filename}"):
                    st.session_state.files_content[filename] = edited_doc
                    st.rerun()
                
                # Add preview toggle
                show_preview = st.toggle("Show Markdown Preview", key=f"preview_toggle_{filename}")
                
                # Display current content
                st.markdown("### Current Documentation:")
                if show_preview:
                    st.code(preview_markdown(st.session_state.files_content[filename]), language="markdown")
                else:
                    st.markdown(st.session_state.files_content[filename])
                
                # Download button
                st.download_button(
                    f"Download {filename} Documentation",
                    st.session_state.files_content[filename],
                    file_name=f"{filename}_documentation.md",
                    mime="text/markdown",
                    key=f"download_{filename}"
                )
                st.markdown("---")
        
        with tab2:
            if st.session_state.readme_content:
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.subheader("README")
                with col2:
                    if st.button("Close", key="close_readme"):
                        st.session_state.readme_content = ""
                        st.rerun()
                
                # Edit box for README
                edited_readme = st.text_area(
                    "Edit README:",
                    value=st.session_state.readme_content,
                    height=400,
                    key="edit_readme"
                )
                
                # Update README content in session state
                if edited_readme != st.session_state.readme_content:
                    st.session_state.readme_content = edited_readme
                
                # Apply Changes button
                if st.button("Apply Changes", key="preview_readme"):
                    st.session_state.readme_content = edited_readme
                    st.rerun()
                
                # Add preview toggle for README
                show_preview = st.toggle("Show Markdown Preview", key="preview_toggle_readme")
                
                # Display current README
                st.markdown("### Current README:")
                if show_preview:
                    st.code(preview_markdown(st.session_state.readme_content), language="markdown")
                else:
                    st.markdown(st.session_state.readme_content)
                
                # Download button
                st.download_button(
                    "Download README.md",
                    st.session_state.readme_content,
                    file_name="README.md",
                    mime="text/markdown",
                    key="download_readme"
                )
    
    # Add a "New Project" button
    if st.session_state.get('generated', False):
        if st.button("Start New Project"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()






