
# RAG PDF Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot that allows you to upload PDF documents and ask questions about their content. Built with Streamlit, LangChain, and Google's Gemini AI.

## 🚀 Features

- **PDF Document Processing**: Upload and process PDF files for intelligent querying
- **RAG Implementation**: Uses Retrieval-Augmented Generation for accurate, context-aware responses
- **Vector Database**: ChromaDB integration for efficient document storage and retrieval
- **Google Gemini AI**: Powered by Google's Gemini 2.0 Flash model for high-quality responses
- **Streamlit Interface**: Clean, user-friendly web interface
- **Chat History**: Maintains conversation context across multiple questions
- **Document Chunking**: Intelligent text splitting for optimal retrieval

## 📋 Prerequisites

- Python 3.8 or higher
- Google AI API key (Gemini)
- Internet connection for API calls

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-pdf-chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   Create a `.env` file in the project root and add your Google AI API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 🔧 Configuration

### Getting a Google AI API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key and add it to your `.env` file

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload a PDF document**
   - Use the file uploader in the sidebar
   - Select a PDF file from your computer
   - Click "Process" to analyze the document

3. **Ask questions**
   - Type your question in the text input field
   - Click "Ask Questions" to get an answer
   - The chatbot will search through the document and provide relevant answers

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── documents/            # Temporary PDF storage
├── db/                   # ChromaDB vector database
│   └── chroma_db_for_rag/
├── modules/              # Additional modules (if any)
└── README.md            # This file
```

## 🔍 How It Works

1. **Document Processing**: PDF files are uploaded and processed using PyPDFLoader
2. **Text Chunking**: Documents are split into smaller chunks using RecursiveCharacterTextSplitter
3. **Vector Embedding**: Text chunks are converted to embeddings using Google's embedding model
4. **Storage**: Embeddings are stored in ChromaDB for efficient retrieval
5. **Query Processing**: When you ask a question:
   - The question is used to retrieve relevant document chunks
   - Relevant chunks are combined with the question
   - Gemini AI generates a response based on the retrieved context
   - Chat history is maintained for context

## 🛡️ Security Notes

- Keep your API keys secure and never commit them to version control
- The `.env` file is automatically ignored by git
- PDF files are temporarily stored and cleaned up between sessions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your Google AI API key is correctly set in the `.env` file
2. **PDF Processing Error**: Ensure the PDF file is not corrupted and is readable
3. **Memory Issues**: Large PDF files may require more memory; consider splitting very large documents
4. **Network Issues**: Ensure you have a stable internet connection for API calls

### Getting Help

If you encounter any issues:
1. Check that all dependencies are installed correctly
2. Verify your API key is valid and has sufficient quota
3. Check the console for error messages
4. Ensure your PDF file is not password-protected

## 🔮 Future Enhancements

- Support for multiple file formats (DOCX, TXT, etc.)
- Batch processing of multiple documents
- Export chat conversations
- Custom embedding models
- Advanced filtering and search options
- User authentication and document sharing

---

**Note**: This application requires an active internet connection to function properly, as it relies on Google's AI APIs for processing and generating responses.
    