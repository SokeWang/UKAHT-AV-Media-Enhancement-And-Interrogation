# 核心技术基线与学术方案 (Baseline Plan)：UKAHT 图像搜推与多模态对齐系统
[English Version (英文版本)](./baseline_plan_en.md)

本基线方案专为数据科学专业毕业设计定制，聚焦于**多模态表示学习（Multimodal Representation Learning）**和**领域自适应（Domain Adaptation）**。系统摒弃了传统的计算机视觉打标或离散拼接，直接采用端到端的双塔表征空间映射，并通过投影适配器微调来优化检索召回精度。

---

## 1. 各技术模块基线设计

### A. 端到端多模态表示对齐 (CLIP & BLIP 双塔基线)
* **目标：** 构建图像与自然语言文本在共享语义空间（Shared Latent Space）中的映射管道。
* **基线算法：**
  - **对比表征学习 (Contrastive Representation Learning)：** 使用预训练的 **CLIP** 模型（`openai/clip-vit-base-patch32`）作为视觉与文本双塔编码器。提取图像和文本查询的 512 维稠密向量（Dense Embeddings），直接通过余弦相似度计算匹配分数。
  - **图像描述生成 (Image Captioning)：** 使用预训练的 **BLIP** 模型（`Salesforce/blip-image-captioning-base`）为图像资产自动生成高精度的自然语言描述，丰富图像的文本语义基础。

### B. 领域自适应投影层 (Domain Adaptation MLP Adapter) —— 数据科学核心创新
* **目标：** 解决通用 CLIP 模型在特定历史遗迹和南极气候图片上的“领域漂移 (Domain Shift)”问题，使向量空间更偏向极地文物和科考场景。
* **算法设计：**
  - 在冻结的 512 维 CLIP 向量层上挂载一个轻量级的多层感知机（MLP）投影适配器（Adapter）：$f_\theta(x)$，将通用向量转换为南极专有的特征表达。
  - **损失函数优化：** 使用**三元组损失（Triplet Loss / Contrastive Loss）**进行微调训练，公式如下：
    $$\mathcal{L} = \max(0, d(a, p) - d(a, n) + \text{margin})$$
    将南极相关图片与对应的口述历史访谈本（Anchor-Positive）在向量空间中拉近，将不匹配的文本/图像（Anchor-Negative）推远。

### C. 多模态加权特征融合检索 (Multimodal Late Fusion)
* **目标：** 结合图像视觉信息与自动描述的语义信息，进行综合语义召回。
* **算法设计：**
  - 提取图像本身的视觉向量（Visual Embedding），以及由 BLIP 生成的文本描述的文本向量（CLIP Text Embedding on Caption）。
  - 在检索推荐时采用晚期线性融合（Weighted Late Fusion）策略：
    $$S_{final} = \alpha \cdot S_{visual} + (1 - \alpha) \cdot S_{textual}$$
    其中 $\alpha$ 为权衡图像特征与文本描述特征的平衡因子。

### D. 检索系统统计评估 (Evaluation Metrics)
* **目标：** 通过标准数据科学指标量化评估算法检索与推荐的精度。
* **评估设计：**
  - 构建含有南极科考特定检索目标的“黄金测试集 (Golden Test Set)”。
  - 对比“通用 CLIP 检索”、“投影 Adapter 对齐检索”以及“多模态特征融合检索”的检索精度，定量计算并对比 **MAP (平均精度均值)** 和 **nDCG (归一化折损累计增益)** 指标。

---

## 2. 依赖项配置 (`backend/requirements.txt`)
```text
# 机器学习与深度学习大模型
transformers>=4.30.0
torch>=2.0.0 --index-url https://download.pytorch.org/ml/cpu

# 矩阵计算与高维向量处理
numpy>=1.24.0
pillow>=9.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
```

---

## 3. 验证与科学实验方案

### 自动化基准验证
- 在 `backend/verify_baselines.py` 中编写基线验证脚本，自动读取测试图片执行多模态特征提取、相似推荐计算、以及多模态向量特征融合，打印功能运行状态。

### 数据科学学术实验 (Jupyter Notebook)
- 编写专门用于训练和评估的 Jupyter Notebook（如 `train_adapter.ipynb`），实现以下实验：
  1. **训练与收敛性分析**：绘制 Triplet Loss 随着 Epoch 的训练收敛曲线。
  2. **特征空间聚类可视化**：利用 **t-SNE / UMAP** 算法将 512 维特征降维为二维空间散点，可视化展示适配器训练前后图像特征的聚类分界线变化。
  3. **召回指标实验**：在测试集上对比不同 $\alpha$ 参数对 nDCG 和 MAP 的影响，并绘制 Precision-Recall 曲线。
