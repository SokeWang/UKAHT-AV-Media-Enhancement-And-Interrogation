# 南极多模态搜推与表征学习系统：6周项目计划 (UKAHT)
[English Version (英文版本)](./project_plan_6_weeks_en.md)

本毕设项目规划书围绕核心学术与工程架构设计：  
**南极图像 $\rightarrow$ 多模态表示模型 (CLIP & BLIP) $\rightarrow$ [领域自适应投影 (MLP Adapter), 特征融合检索 (Late Fusion), 检索统计评估 (MAP/nDCG)]**

为应对项目周期与可行性风险，系统采用 **“快速跑通简单基线（CLIP/BLIP Baseline），随后迭代领域微调与智能 Agent”** 的渐进式开发路线，并将每个里程碑的交付产物与团队分工进行明确绑定。

---

### **团队分工与职责 (Team Organisation)**

* **王沛东 (Peidong Wang)**
  * **学术职责**：领域自适应与多模态表征（设计实现自定义 MLP Adapter 层，三元组损失函数，解决极地环境领域漂移）。
  * **工程职责**：后端基础设施与 API 逻辑（基于 FastAPI 的异步 RESTful API 研发，封装底层 model 推理，提供稳定的 RAG 接口）。
* **罗天 (Tian Luo)**
  * **学术职责**：LLM Agent 与智能 RAG 架构（设计 ReAct 提示词策略，实现 LLM 工具调用和意图解析）。
  * **工程职责**：混合数据库与检索引擎（Milestone 1 搭建 SQLite 原型，Milestone 2 优化升级切换至 PostgreSQL 数据库，实现向量 + SQL 元数据混合查询与索引优化）。
* **张艺升 (Yisheng Zhang)**
  * **学术职责**：自动处理与特征工程（部署 BLIP 批量生成 Caption，执行 3.3 GB 图像数据导入，提取高维特征向量）。
  * **工程职责**：容器化部署与 DevOps（编写 Dockerfile 构建环境隔离，配置 docker-compose 容器编排）。
* **袁晨宇 (Chenyu Yuan)**
  * **学术职责**：学术评估与高维可视化（编写评估脚本计算 MAP 和 nDCG 指标，实现 t-SNE / UMAP 特征降维图）。
  * **工程职责**：Interactive UI 开发（ Milestone 1 使用 Streamlit 搭建基础原型；Milestone 2 优化升级切换至 React SPA，构建多轮对话检索界面，图表和推荐卡片展示，开发交互式人工标注 UI）。

---

### **6周日程表**

#### **里程碑 1：核心 Pipeline 基线与原型验证（第 1-2 周）**
*目标：快速搭建出前后端打通的简单模型检索 Pipeline，验证可行性。*
* **交付产物 (Explicit Outputs)**：
  1. **SQLite 种子数据库**：在 `backend/db.sqlite` 中建立初始表结构，存储图像元数据、BLIP 描述和 512 维特征向量，填充种子数据。
  2. **后端 FastAPI 服务**：实现 `/api/search`（文本语义检索）和 `/api/recommend`（相似图推荐）接口。
  3. **前端 Streamlit 原型界面**：完成 Streamlit 检索网格布局与相似推荐面板连接。
  4. **自动化验证脚本**：`backend/verify_baselines.py` 能够成功执行并打印余弦相似度分数。
* **团队分工**：
  * **王沛东**：封装预训练 CLIP & BLIP 推理接口，构建 FastAPI 服务。
  * **罗天**：设计并初始化 SQLite 种子数据库。
  * **袁晨宇**：完成前端 Streamlit 原型界面搭建与接口连接。

#### **里程碑 2：架构优化升级（切换至 PostgreSQL + React SPA）与全量数据索引（第 3 周）**
*目标：全量处理 3.3 GB 图像，完成数据库与前端架构升级，构建系统的精度评估基线。*
* **交付产物 (Explicit Outputs)**：
  1. **PostgreSQL 全量图像数据库**：将数据库优化升级迁移至 PostgreSQL 容器（`ukaht-db`），全量导入 3.3 GB 图像特征向量及 BLIP 自动生成的描述。
  2. **React SPA 架构与辅助标注 UI**：弃用 Streamlit 原型，全面升级为 React 单页应用（SPA），并集成人工介入（Human-in-the-loop）的辅助打标与校对界面。
  3. **人工标注黄金测试集**：导出为 `golden_test_set.json` 文件，包含 50–100 对代表性的“强正样本”图像-文本查询对。
* **团队分工**：
  * **张艺升**：部署 BLIP 批量推理管道，运行全局数据导入脚本将 3.3 GB 图像特征存入 PostgreSQL。
  * **罗天**：完成从 SQLite 到 PostgreSQL 的数据库架构升级迁移，优化批量提交事务与索引结构。
  * **袁晨宇**：完成前端 UI 从 Streamlit 到 React SPA 的重构与辅助标注页面开发。

#### **里程碑 3：极地领域自适应 MLP 适配器开发（第 4 周）**
*目标：通过对比学习训练 MLP 层，纠正 CLIP 模型在极地文物和科考场景下的领域漂移。*
* **交付产物 (Explicit Outputs)**：
  1. **三元组训练管道**：实现可输出（Anchor, Positive, Negative）三元组数据批次的 Python 生成器。
  2. **适配器 model 文件**：训练完成并导出的 PyTorch 适配器参数文件 `adapter.pth`。
  3. **训练过程记录 Notebook**：`train_adapter.ipynb` 展示 Triplet Loss 随 Epoch 递减的收敛曲线。
* **团队分工**：
  * **王沛东**：设计 2 层 MLP 投影头并编写自定义三元组损失函数（Triplet Loss），训练适配器并集成到 backend 推理中。
  * **张艺升**：开发正负样本对随机采样算法，用于生成三元组数据集。

#### **里程碑 4：LLM Agent 与 RAG 管道集成（第 5 周）**
*目标：结合大模型 Tool-Calling 与 RAG 架构，支持复杂自然语言多轮问答。*
* **交付产物 (Explicit Outputs)**：
  1. **Agent 路由核心**：基于 ReAct 架构的 Prompt 模板与 Python 查询意图路由逻辑，将自然语言自动转换为结构化 SQL + 向量的混合查询。
  2. **RAG 上下文合成引擎**：将检索到的多模态结果与历史对话融合，调用 LLM 生成问答文本。
  3. **多轮对话 UI**：在前端实现抽屉式多轮 Chat 面板。
* **团队分工**：
  * **罗天**：设计大模型的 Prompt 策略，开发 Tool-calling 混合检索工具。
  * **王沛东**：提供后端 RAG 推理 API 接口，并对接 Adapter 空间特征提取。
  * **袁晨宇**：集成前端多轮对话交互界面。

#### **里程碑 5：学术精度评估与 DevOps 部署（第 6 周）**
*目标：对检索算法做消融实验，分析特征空间变化，并实现一键交付部署。*
* **交付产物 (Explicit Outputs)**：
  1. **检索精度对比表**：在 Notebook 中评测并打印“Raw CLIP 检索” vs “Adapter 对齐检索” vs “多模态特征融合检索”的 **MAP** 和 **nDCG** 精度的对比数据表格。
  2. **特征空间降维散点图**：使用 t-SNE / UMAP 算法将 512 维特征降为 2D 散点图，直观展现适配器微调前后的图像聚类分界变化。
  3. **Docker 编排配置**：编写 `Dockerfile` 与 `docker-compose.yml` 配置文件，打包后端服务（其中静态托管了编译后的 React 前端，避免了额外的 Streamlit 前端容器）。
  4. **评估展示看板**：在前端 React 界面集成静态 Evaluation 展示页面（原定在 Streamlit 界面集成，后由于定制化需要转至 React），读取并展示离线评测得到的精度对比表与 t-SNE 散点图。
* **团队分工**：
  * **袁晨宇**：编写 MAP/nDCG 评估脚本，绘制 t-SNE / UMAP 聚类图，集成前端 React 静态评估展示界面。
  * **张艺升**：编写 Dockerfile 进行容器化打包并静态托管编译后的 React SPA 资源，进行跨平台依赖管理。
