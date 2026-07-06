# UKAHT Antarctic Media Interrogation Portal: Multimodal Search & Recommendation System
[中文版本 (Chinese Version)](./README_zh.md)

An end-to-end multimodal representation retrieval and recommendation system designed for the UK Antarctic Heritage Trust (UKAHT) historical archives. The system leverages contrastive vision-language models (CLIP) and generative captioning models (BLIP) to align polar media and oral history transcripts, enabling natural language search and visual recommendation capabilities.

---

## 🛠️ Cross-Platform Developer Guidelines (Crucial)

Since team members develop on both **macOS** and **Windows**, please strictly adhere to the following rules:

### 1. File Path Resolution (No Hardcoding)
*   **Rule:** **Never** write hardcoded slashes (like `folder/file.jpg` or `C:\\folder\\file`) in Python code.
*   **Usage:** Always use `os.path.join` or Python's `pathlib.Path` to construct file and directory paths.
    ```python
    # Correct:
    filepath = os.path.join("backend", "static", "images", "sample.png")
    
    # Incorrect:
    filepath = "backend/static/images/sample.png"
    ```

### 2. Git Line Endings
Ensure git handles line endings properly across Windows (CRLF) and Mac/Linux (LF):
```bash
# Windows users run this in git bash:
git config --global core.autocrlf true

# macOS/Linux users run this in terminal:
git config --global core.autocrlf input
```

## 🚀 Docker Compose Deployment (Recommended & Only Supported Method)

The application is fully containerized. To ensure dependency alignment and prevent cross-platform configuration errors, **Docker Compose is the only supported deployment method**.

### 1. Prerequisites
*   Ensure **Docker** and **Docker Compose** are installed and running.
*   (Optional for Agent) Start Ollama on your host machine and pull your LLM (e.g., `ollama run gemma2`) to enable RAG.

### 2. Start the System
From the repository root, run:
```bash
docker compose up --build -d
```

This will automatically build and launch both backend and frontend microservices:
*   **FastAPI Backend API**: Accessible at `http://localhost:8000` (docs at `http://localhost:8000/docs`).
*   **Streamlit Frontend UI**: Open your browser at `http://localhost:8501` to use the portal.

### 3. Stop the System
```bash
docker compose down
```

---

## 🔍 Log Monitoring & Maintenance
To check service logs or troubleshoot:
```bash
# View backend container logs
docker compose logs backend --tail 50 -f

# View frontend container logs
docker compose logs frontend --tail 50 -f
```
