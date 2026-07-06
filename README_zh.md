# UKAHT 南极视听媒体智能问答与搜推系统
[English Version (英文版本)](./README.md)

基于双塔表示学习（Contrastive Representation Learning）构建的端到端多模态图像语义检索与推荐系统，专为英国南极遗产信托基金会（UKAHT）历史图像档案整理设计。系统集成了 CLIP 与 BLIP 大模型，实现了历史照片的自然语言交互搜索、口述访谈本语义关联匹配，以及高维特征的图像相似度关联推荐。

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

## 🚀 Docker Compose 一键部署 (唯一支持的运行方式)

为保证跨平台依赖一致性、避免繁琐的环境配置，本系统**仅支持通过 Docker Compose 容器化部署**，不再保留本地手动安装部署方式。

### 1. 前置准备
*   确保您的宿主机上已经安装并启动了 **Docker** 与 **Docker Compose**。
*   （可选，用于 AI Agent）在宿主机启动 Ollama 并拉取大模型（如 `ollama run gemma2`）。

### 2. 启动系统
在项目根目录下，执行以下命令：
```bash
docker compose up --build -d
```

该命令将自动下载基础镜像、构建服务并以守护进程模式启动前后端微服务：
*   **后端 FastAPI 服务**：访问地址为 `http://localhost:8000`（API 交互文档地址：`http://localhost:8000/docs`）。
*   **前端 Streamlit 界面**：在浏览器中打开 **`http://localhost:8501`** 即可开始使用智能搜推与标注功能。

### 3. 停止系统
```bash
docker compose down
```

---

## 🔍 日志监控与维护
若需查看运行状态或排查故障，可运行以下命令：
```bash
# 查看后端运行日志
docker compose logs backend --tail 50 -f

# 查看前端运行日志
docker compose logs frontend --tail 50 -f
```
