# 🤖 Agentic RAG Development Pipeline

An AI-powered Streamlit application that automates software development workflows using Retrieval-Augmented Generation (RAG) and multiple specialized AI agents. Transform RFP documents into complete development artifacts including user stories, system design, code, and test cases.

## 🌟 Features

- **Intelligent RFP Processing**: Upload or reference RFP documents to extract requirements
- **Multi-Agent Architecture**:
  - Business Analyst Agent: Generates detailed user stories with story points
  - Design Agent: Creates software architecture and design specifications
  - Tech Stack Detection: Automatically identifies appropriate technologies
  - Developer Agent: Produces clean, modular code
  - Tester Agent: Generates comprehensive test cases
- **Vector Search**: Chroma-based semantic search for context retrieval
- **Real-time Generation**: Interactive Streamlit interface for immediate results
- **Modular Code Output**: Clean, production-ready code snippets

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Google AI API key (for Gemini model)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/IshanChattopadhaya/GenAI.git
   cd GenAI
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Google API key:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Replace `<API_KEY>` in `app.py` with your actual key

## 📋 Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to the provided URL (usually `http://localhost:8501`)

3. Upload or reference an RFP document (default: `rfp.txt`)

4. Click through the pipeline:
   - **Load RFP**: Process the document and create vector store
   - **Business Analysis**: Generate user stories and estimates
   - **System Design**: Create architecture specifications
   - **Tech Stack**: Detect appropriate technologies
   - **Code Generation**: Produce implementation code
   - **Testing**: Generate test cases

## 🛠️ Requirements

- streamlit
- langchain
- langchain-community
- langchain-google-genai
- chromadb
- sentence-transformers
- python-dotenv

## ⚙️ Configuration

- **CHROMA_DIR**: Directory for Chroma vector database (default: `chroma_pipeline`)
- **RFP_PATH**: Path to RFP document (default: `rfp.txt`)
- **REPO_DIR**: Directory for generated code (default: `repo`)
- **Model**: Gemini 2.0 Flash (configurable in `load_llm()`)

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── rfp.txt               # Sample RFP document
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
├── chroma_pipeline/     # Vector database (ignored)
├── repo/                # Generated code output
└── README.md            # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔧 Troubleshooting

- **API Key Issues**: Ensure your Google API key is valid and has sufficient quota
- **Dependencies**: Run `pip install -r requirements.txt` if you encounter import errors
- **Vector Store**: Delete `chroma_pipeline/` directory if you need to rebuild the database

## 📞 Support

For questions or issues, please open an issue on GitHub.
