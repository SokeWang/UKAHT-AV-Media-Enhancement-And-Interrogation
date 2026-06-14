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

---

## 🚀 How to Run the App Locally

### 1. Start Backend Server
Inside the repository root directory:
```bash
# Install dependencies
pip3 install -r backend/requirements.txt

# Run the FastAPI server with hot-reload enabled
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
The interactive API documentation is accessible at `http://127.0.0.1:8000/docs`.

### 2. Start Frontend Dev Server
Inside the `frontend` folder:
```bash
# Install dependencies
npm install

# Run the React hot-reload Vite server
npm run dev
```
Open **`http://localhost:5173`** in your browser. All API requests to `/api` and `/static` are automatically proxied to the FastAPI server running on port `8000`.

---

## 📦 Universal Docker Packaging (For Final Handoff)
The application can be bundled into a CPU-compatible Docker container, ensuring it runs seamlessly on any host system:
```bash
docker-compose up --build
```
