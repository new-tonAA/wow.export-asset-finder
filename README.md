# WoW Asset Finder

> CLIP + FAISS 驱动的魔兽世界模型资产相似度搜索工具

![WoW Asset Finder Screenshot](images/Readme.png)

---

## 功能特性

- **图片搜索** — 上传/粘贴一张参考图，找出最相似的游戏模型
- **文字搜索** — 输入英文描述（如 "stone bridge"、"wooden barrel"）直接搜索
- **配色搜索** — 按颜色分布匹配，找风格配色相近的模型
- **混合搜索** — 图片 + 文字加权组合
- **实时缩略图** — 搜索结果实时从 wow.export 渲染预览（不存储）
- **分类提取** — 按 WMO/Creatures/Items 等类别独立提取，支持断点续传

---

## 搜索模式

| 模式 | 输入 | 说明 |
|------|------|------|
| **Semantic** | 图片 | CLIP 语义匹配，找"看起来是同类东西"的模型 |
| **Color** | 图片 | HSV 颜色直方图匹配，找"配色相似"的模型 |
| **Text** | 文字 | CLIP 文本编码，用自然语言描述搜索 |
| **Combined** | 图片 + 文字 | 多维度加权融合 |

---

## 快速开始

### 1. 安装

```bash
git clone https://github.com/new-tonAA/wow.export-asset-finder.git
cd wow.export-asset-finder
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> PyTorch GPU 加速请按 https://pytorch.org 选择对应 CUDA 版本

### 2. 前置条件

- Python 3.10+
- [wow.export](https://github.com/Kruithne/wow.export) debug build 放在 `C:\wow.export\wow.export\bin\win-x64-debug\`
- 能访问暴雪 CDN（建议选 Taiwan 区域）

### 3. 启动

**Windows：** 双击 `start.vbs`

**手动：**
```bash
venv\Scripts\python app.py
```

启动后约 40 秒加载 CLIP 模型，完成后自动打开浏览器 http://localhost:5001

### 4. 使用

1. **Connect** — 选择区域和版本，连接 wow.export 到暴雪 CDN
2. **Extract** — 选择模型类别，点击 Extract All 提取特征向量
3. **Search** — 上传图片 / Ctrl+V 粘贴 / 输入文字描述

---

## 模型类别

| 类别 | 内容 | 数量 |
|------|------|------|
| WMO - Buildings & Structures | 城堡、教堂、桥梁、塔楼 | ~12,000 |
| Creatures | 生物模型 | - |
| Items | 物品模型 | - |
| Characters | 角色模型 | - |
| Doodads & Props | 桶、栅栏、灯具等 | - |
| Environment | 岩石、树木 | - |
| Spells & Effects | 法术特效 | - |
| All Models | 全部 | 143,000+ |

---

## 项目结构

```
wow-asset-finder/
├── app.py                  # Web 后端 (Flask + SocketIO)
├── extract_features.py     # 模型加载、截图、特征提取
├── search.py               # CLI 搜索（可选）
├── templates/index.html    # 前端页面
├── splash.ps1              # 启动加载画面
├── start.vbs               # Windows 启动入口
├── requirements.txt        # Python 依赖
├── features/               # 特征向量（按类别存储，gitignore）
└── wow_export_data/        # CASC 缓存（自动生成，gitignore）
```

---

## 技术栈

- **OpenCLIP** (ViT-B-32) — 图像/文本特征提取
- **FAISS** — 向量相似度搜索
- **Flask + SocketIO** — Web 服务 + 实时通信
- **Selenium** — 自动化控制 wow.export (nw.js)

---

## 注意事项

- wow.export 一次只能运行一个实例
- 首次连接需从 CDN 下载数据（较慢），后续有缓存
- 提取 WMO 全部模型约 8-17 小时，支持断点续传（Resume）
- 搜索不需要连接 wow.export，有 features 文件即可
- 不同类别的特征独立存储，互不覆盖
