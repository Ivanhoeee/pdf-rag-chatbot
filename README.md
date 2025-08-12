# RAG Explorer 📚

A Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with PDF documents using various language models.

## Features

- 📄 Upload and process PDF documents
- 🤖 Chat with your documents using multiple models (Flan-T5, GPT-3.5, GPT-4)
- 🎯 View embedding visualizations to understand document structure
- 🔍 See the RAG process in action with detailed explanations

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Ivanhoeee/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key (if using GPT models)
OPENAI_API_KEY=your_actual_api_key_here
```

### 5. Run the application
```bash
streamlit run app.py
```

## Usage

1. **Upload a PDF**: Use the file uploader to select your document
2. **Ask questions**: Type questions about your document in the chat interface
3. **Explore embeddings**: Switch to the "Visualize Embeddings" tab to see how your document and questions relate semantically
4. **Change models**: Use the sidebar to switch between different language models

## Models Supported

- **Flan-T5 Small**: Free, local model good for basic tasks
- **GPT-3.5 Turbo**: OpenAI's fast and capable model (requires API key)
- **GPT-4**: OpenAI's most advanced model (requires API key)

## Security Note

- Never commit your `.env` file to version control
- Keep your API keys secure and don't share them
- The `.env` file is automatically ignored by git

## Contributing

Pull requests are welcome! Please ensure you don't commit any API keys or sensitive information.

## License

This project is open source and available under the MIT License.
