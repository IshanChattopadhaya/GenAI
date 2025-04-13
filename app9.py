import streamlit as st
import os
import shutil
import re
import tempfile
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

# --- Config ---
CHROMA_DIR = "chroma_pipeline"
RFP_PATH = "rfp.txt"
REPO_DIR = "repo"
os.environ["GOOGLE_API_KEY"] = "AIzaSyC7LGeldf_RC79PMJVq5TLLd2fN5aMA8Io"
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(REPO_DIR, exist_ok=True)

# --- Title ---
st.title("🤖 Agentic RAG Development Pipeline")

# --- Load Models ---
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

embedder = load_embedder()
llm = load_llm()

# --- Helpers ---
def create_vector_store(path):
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    loader = TextLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    db = Chroma.from_documents(split_docs, embedder, persist_directory=CHROMA_DIR)
    db.persist()
    return db

def get_context(db, query, k=3):
    docs = db.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])

def business_agent(rfp_context):
    prompt = (
        "You are a Business Analyst. Extract detailed user stories from the RFP context below.\n\n"
        f"Context:\n{rfp_context}\n\n"
        "For each user story, use the format: 'As a [user], I want to [action] so that [benefit]'.\n"
        "Also, assign an estimated story point to each user story using the Fibonacci scale (1, 2, 3, 5, 8, 13, ...).\n"
        "After listing the user stories, provide a summary estimation table like:\n\n"
        "| User Story # | Story Points |\n"
        "|--------------|--------------|\n"
        "| 1            | 3            |\n"
        "| 2            | 5            |\n"
        "| ...          | ...          |\n\n"
        "Then, summarize the total effort by summing the story points.\n"
    )
    return llm.invoke(prompt).content.strip()

def design_agent(user_stories):
    prompt = (
        "You are a Software Designer. Based on the following user stories, create a software design outline.\n\n"
        f"User Stories:\n{user_stories}\n\n"
        "Include modules, data flow, and architecture decisions."
    )
    return llm.invoke(prompt).content.strip()

def detect_tech_stack(rfp_context):
    prompt = (
        "You are a technical analyst. Read the following RFP context and extract the main technology stack "
        "(e.g., Python, JavaScript, Node.js, Java, etc.). Return only the tech stack as plain text. "
        "Avoid explanation.\n\n"
        f"RFP Context:\n{rfp_context}"
    )
    return llm.invoke(prompt).content.strip().lower()

def developer_agent(user_stories, design_notes, tech_stack):
    prompt = (
        f"You are a Developer. Based on the user stories and design notes below, write modular and readable code using {tech_stack}.\n\n"
        f"User Stories:\n{user_stories}\n\nDesign Notes:\n{design_notes}\n\n"
        f"Generate clean code only. Avoid explanation."
    )
    return llm.invoke(prompt).content.strip()

def sanitize_code(code):
    sanitized = re.sub(r'[^\x00-\x7F]+', '', code)              # Remove non-ASCII
    sanitized = sanitized.replace("\u200b", "").strip()         # Remove zero-width spaces
    sanitized = re.sub(r"^```(?:\w+)?", "", sanitized)          # Remove opening triple backticks with optional language
    sanitized = re.sub(r"```$", "", sanitized)                  # Remove closing triple backticks
    return sanitized.strip()

def tester_agent(user_stories, code, tech_stack):
    sanitized_code = sanitize_code(code)
    if "python" in tech_stack:
        test_instruction = "using Python’s `unittest` framework"
    elif "javascript" in tech_stack or "node.js" in tech_stack:
        test_instruction = "using Mocha and Chai for Node.js"
    else:
        test_instruction = "using the standard testing framework of the specified language"

    prompt = (
        f"You are a Tester. Based on the user stories below, generate 4 to 5 simple test cases in {tech_stack} "
        f"{test_instruction}. Avoid using any external libraries beyond what's standard. "
        f"Return ONLY the code without explanation or markdown.\n\n"
        f"User Stories:\n{user_stories}\n\nCode:\n{sanitized_code}"
    )
    return sanitize_code(llm.invoke(prompt).content.strip())

def simulate_test_results(code, test_code):
    prompt = (
        "You are a senior QA engineer. Given the code and its corresponding test cases below, analyze the logic and return simulated test results.\n\n"
        f"Code:\n{code}\n\nTest Cases:\n{test_code}\n\n"
        "For each test case, determine if it would likely pass or fail. Return results in this format:\n"
        "TestFunctionName: PASSED/FAILED - [Reason if failed]"
    )
    return llm.invoke(prompt).content.strip()

def project_manager_agent(user_stories, design, code, tests):
    prompt = (
        "You are a Project Manager overseeing the given project. Summarize the overall progress from the following artifacts:\n\n"
        f"User Stories:\n{user_stories}\n\n"
        f"Design:\n{design}\n\n"
        f"Code:\n{code[:300]}...\n\n"
        f"Tests:\n{tests}\n\n"
        "Provide a professional project summary."
    )
    return llm.invoke(prompt).content.strip()

# --- Upload RFP ---
uploaded_file = st.file_uploader("📄 Upload RFP Document", type=["txt"])

# --- Session State Setup ---
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False

# --- On Upload ---
if uploaded_file and not st.session_state.pipeline_run:
    with open(RFP_PATH, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ RFP uploaded successfully!")

    with st.status("🛠️ Running Agentic Pipeline...", expanded=True) as status:
        st.write("🔍 Creating vector store...")
        db = create_vector_store(RFP_PATH)

        st.write("📖 Extracting RFP context...")
        rfp_context = get_context(db, "project requirements")

        st.write("🔎 Detecting tech stack...")
        tech_stack = detect_tech_stack(rfp_context)

        st.write("🧠 Business Analyst is generating user stories with estimation...")
        user_stories = business_agent(rfp_context)

        st.write("📐 Designer is creating architecture and modules...")
        design_notes = design_agent(user_stories)

        st.write(f"💻 Developer is generating {tech_stack} code...")
        code = developer_agent(user_stories, design_notes,tech_stack)

        st.write("🧪 Tester is generating test cases...")
        test_code = tester_agent(user_stories, code, tech_stack)

        st.write("🧾 Simulating test execution...")
        test_results = simulate_test_results(code, test_code)

        st.write("📋 Project Manager is summarizing progress...")
        pm_summary = project_manager_agent(user_stories, design_notes, code, test_code)

        status.update(label="✅ Pipeline Complete!", state="complete")

    st.session_state.update({
        "user_stories": user_stories,
        "design_notes": design_notes,
        "code": code,
        "test_code": test_code,
        "test_results": test_results,
        "pm_summary": pm_summary,
        "pipeline_run": True
    })
    st.rerun()

# --- Display Results ---
if st.session_state.pipeline_run:
    st.subheader("✅ Pipeline Outputs")

    tabs = st.tabs(["🧠 User Stories", "🧹 Design", "💻 Code", "🧪 Test Cases", "✅ Test Results", "📋 Summary"])
    with tabs[0]:
        st.text_area("📋 User Stories", st.session_state.user_stories, height=300)
    with tabs[1]:
        st.text_area("📐 Design", st.session_state.design_notes, height=300)
    with tabs[2]:
        st.text_area("📂 Generated Code", st.session_state.code, height=400)
        st.download_button("⬇️ Download Code", st.session_state.code, file_name="generated_code.py")
    with tabs[3]:
        st.text_area("🧪 Test Cases", st.session_state.test_code, height=400)
    with tabs[4]:
        st.text_area("✅ Test Execution Results", st.session_state.test_results, height=400)
    with tabs[5]:
        st.success(st.session_state.pm_summary)

    st.divider()

    st.subheader("🔎 Ask Anything About the Project")
    query = st.chat_input("Ask the project assistant...")

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as tmp_dir:
                docs = [
                    Document(page_content=st.session_state.user_stories, metadata={"type": "User Stories"}),
                    Document(page_content=st.session_state.design_notes, metadata={"type": "Design"}),
                    Document(page_content=st.session_state.code, metadata={"type": "Code"}),
                    Document(page_content=st.session_state.test_code, metadata={"type": "Tests"}),
                    Document(page_content=st.session_state.pm_summary, metadata={"type": "Summary"}),
                ]
                query_db = Chroma.from_documents(docs, embedding=embedder)
                results = query_db.similarity_search(query, k=6)
                context = "\n\n".join([f"{doc.metadata['type']}:\n{doc.page_content}" for doc in results])
                rag_prompt = (
                    f"You are an expert AI assistant. Use the following project artifacts to answer the user’s question.\n\n"
                    f"Project Artifacts:\n{context}\n\n"
                    f"User Question:\n{query}\n\n"
                    f"Answer based only on the provided context. "
                    f"If the information is not available, reply with 'The information is not available in the provided context.' "
                    f"Do not make assumptions. Be specific and detailed in your response."
                )
                answer = llm.invoke(rag_prompt).content.strip()
                st.markdown(answer)
else:
    st.info("📅 Please upload a single RFP document to begin.")