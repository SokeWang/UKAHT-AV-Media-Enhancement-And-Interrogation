# UKAHT 南极视听媒体增强与问答系统
[English Version (英文版本)](./README.md)

自动分类、元数据富集及计算机视觉工具，用于英国南极遗产信托基金会（UKAHT）历史南极媒体库的整理、站点监测与遗迹保护分析。

---

## 🛠️ 跨平台开发规范（至关重要）

由于小队成员同时使用 **macOS** 和 **Windows** 进行开发，请严格遵守以下开发规范：

### 1. 文件路径解析（禁止硬编码）
*   **规则：** **绝对不要**在 Python 代码中硬编码写死斜杠或反斜杠（例如 `folder/file.jpg` 或 `C:\\folder\\file`）。
*   **做法：** 务必使用 `os.path.join` 或 Python 的 `pathlib.Path` 来拼接和处理文件与目录路径。
    ```python
    # 正确的做法：
    filepath = os.path.join("backend", "static", "images", "sample.png")
    
    # 错误的做法：
    filepath = "backend/static/images/sample.png"
    ```

### 2. Git 换行符配置
确保 Git 能够正确处理 Windows（CRLF）与 Mac/Linux（LF）之间的换行符转换，避免提交时产生大面积冲突：
```bash
# Windows 开发者请在 Git Bash 中运行：
git config --global core.autocrlf true

# macOS/Linux 开发者请在终端中运行：
git config --global core.autocrlf input
```

### 3. 本地 COLMAP 配置（用于 3D 重建）
后端会通过系统命令直接调用 COLMAP。请确保其已安装并加入系统环境变量：
*   **macOS：** 在终端运行 `brew install colmap` 进行安装。
*   **Windows：** 从 [COLMAP GitHub Releases](https://github.com/colmap/colmap/releases) 下载编译好的 Windows 压缩包，解压后，将包含 `colmap.exe` 的 `bin` 文件夹路径添加至系统的环境变量 `Path` 中。

---

## 🚀 本地运行 MVP 步骤

### 1. 启动后端 API 服务
在项目根目录下：
```bash
# 安装 Python 依赖库
pip3 install -r backend/requirements.txt

# 启动 FastAPI 服务并开启热重载 (--reload)
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
启动后可通过浏览器访问 `http://127.0.0.1:8000/docs` 查看交互式 API 文档。

### 2. 启动前端开发服务器
在 `frontend` 目录下：
```bash
# 安装前端依赖包
npm install

# 启动 React 热更新开发服务
npm run dev
```
启动后通过浏览器访问 **`http://localhost:5173`**。前端对 `/api` 和 `/static` 的所有请求都已配置反向代理（Proxy），会自动转发到运行在 `8000` 端口的 FastAPI 后端服务。

---

## 📦 通用 Docker 打包（用于终期交付）
在项目后期，整个应用将被打包成基于 CPU 的通用 Linux Docker 容器，确保客户（UKAHT 工作人员）可以在任何宿主机系统上一键运行：
```bash
docker-compose up --build
```
