# RAG Explained with Code

## Project Structure

```
rag-explained-with-code/
│
├── data/
│   └── products.json
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── embeddings.py
│   ├── vector_db.py
│   ├── llm_router.py
│   └── rag_pipeline.py
│
├── tests/
│   ├── __init__.py
│   └── test_rag_pipeline.py
│   └── test_openai_api.py
│
├── pyproject.toml
├── README.md
├── requirements.txt
└── product_assistant.py
```

## Overview
A Retrieval-Augmented Generation (RAG) system for product recommendations using OpenAI embeddings, FAISS vector search, and GPT for intelligent responses.

## Prerequisites
- Python 3.8+
- uv (Universal Python Package Manager)
- OpenAI API Key

## Installation

1. Clone the repository
```bash
git clone https://github.com/praveenkannan/rag-explained-with-code.git
cd rag-explained-with-code
```

2. Install uv (if not already installed)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
irm https://astral.sh/uv/install.ps1 | iex
```

3. Create and activate virtual environment
```bash
uv venv  # Create virtual environment
source .venv/bin/activate  # Activate (Unix)
# Or on Windows
.venv\Scripts\activate
```

4. Install dependencies
```bash
uv pip install -e .
```

5. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Running the Application

### Interactive Mode
```bash
uv run product-assistant
# Or
python product_assistant.py
```

### Single Query Mode
```bash
uv run product-assistant "What chair is best for back pain?"
# Or
python product_assistant.py "What chair is best for back pain?"
```

## Testing

### Running Tests
```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run tests and show print statements
uv run pytest -s

# Run tests and generate coverage report
uv run pytest --cov=src
```

### Test Coverage
The test suite covers:
- Catalog loading
- Product filtering
- Embedding generation
- Vector database functionality
- Answer generation
- Product addition

### Specific Test Runs
```bash
# Run a specific test file
uv run pytest tests/test_rag_pipeline.py

# Run a specific test function
uv run pytest tests/test_rag_pipeline.py::TestRAGProductAssistant::test_catalog_loading
```

### Testing OpenAI API Key

To verify your OpenAI API key:

1. Run the API key test script:
```bash
python tests/test_openai_api.py
```

Possible Test Outcomes:
- **Success**: API key is valid and functional
- **Failure**: Detailed error messages will guide you

Common API Key Issues:
- Incorrect or expired key
- Billing or account restrictions
- Rate limit exceeded
- Network connectivity problems

#### Troubleshooting API Key
1. Verify key at [OpenAI Platform](https://platform.openai.com/account/api-keys)
2. Check account billing status
3. Ensure network connectivity
4. Reset API key if necessary

## Development

### Linting
```bash
uv run ruff check .
```

### Code Formatting
```bash
uv run ruff format .
```

## Troubleshooting
- Ensure `.env` file is correctly configured with OpenAI API key
- Check that all dependencies are installed
- Verify Python version compatibility

## Troubleshooting and Reset

### Resetting the Project
If you encounter issues or want to start from scratch:

1. Run the reset script:
```bash
./reset_project.sh
```

2. Update `.env` with your OpenAI API key:
```bash
nano .env
# Replace 'your_openai_api_key_here' with your actual OpenAI API key
```

### Common Issues
- **API Key Errors**: 
  - Ensure you have a valid OpenAI API key
  - Check https://platform.openai.com/account/api-keys
  - Verify you have sufficient API credits

- **Virtual Environment Problems**:
  - Always activate the virtual environment before running
  ```bash
  source .venv/bin/activate  # Unix
  # Or on Windows
  .venv\Scripts\activate
  ```

- **Dependency Issues**:
  ```bash
  # Reinstall dependencies
  uv pip install -e .
  uv pip install -e .[dev]
  ```

## License
MIT License

## Technologies
- Python
- OpenAI API
- FAISS
- NumPy
- Pytest

## Author
[Praveen Kannan](https://github.com/praveenkannan/rag-explained-with-code)
