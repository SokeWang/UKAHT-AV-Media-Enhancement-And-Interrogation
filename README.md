# UKAHT-AV-Media-Enhancement-And-Interrogation
Automated classification, metadata enrichment, and computer vision tools for UK Antarctic Heritage Trust's historical Antarctic media library, site monitoring, and conservation analysis.

---

## 🛠️ Cross-Platform Developer Guidelines (Crucial)

Since team members are developing on both **macOS** and **Windows**, please strictly adhere to the following rules:

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

### 3. Native COLMAP Setup (for 3D Reconstruction)
The backend calls COLMAP as a system command. Make sure it is installed and added to your path:
*   **macOS:** Run `brew install colmap` in your terminal.
*   **Windows:** Download the binary zip package from [COLMAP Releases](https://github.com/colmap/colmap/releases). Unpack it, and add the path to the folder containing `colmap.exe` to your Windows user Environment Variables (`Path`).

---

## 🚀 How to Run the MVP Locally

### 1. Start Backend Server
Inside the repository root directory:
```bash
# Install dependencies
pip3 install -r backend/requirements.txt

# Run the FastAPI server with hot-reload enabled
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
The API documentation is accessible at `http://127.0.0.1:8000/docs`.

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
In the final stages, the application will be bundled into a CPU-compatible Docker container. This ensures that the client (UKAHT staff) can deploy it easily on any host system:
```bash
docker-compose up --build
```
