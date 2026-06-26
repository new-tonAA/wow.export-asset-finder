# WoW Asset Finder

通过图片/文字相似度搜索魔兽世界模型资产，缩短查找资源的时间。

## 工作原理

1. 通过 Selenium 自动化控制 [wow.export](https://github.com/Kruithne/wow.export)（nw.js 应用），从暴雪 CDN 实时拉取模型并截取 3D 预览图
2. 用 CLIP 模型提取语义特征向量 + HSV 颜色直方图，按类别存储为 `.npz` 文件
3. 搜索时加载所有类别的特征向量，通过 FAISS 做最近邻搜索返回最相似的 12 个模型

## 搜索模式

| 模式 | 说明 |
|------|------|
| Semantic | CLIP 图搜图，找"看起来是同类东西"的模型 |
| Color | HSV 颜色直方图匹配，找"配色相似"的模型 |
| Text | 输入文字描述（英文），如 "stone bridge"、"wooden barrel" |
| Combined | 图片 + 文字加权融合搜索 |

## 安装

```bash
cd wow-asset-finder
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

PyTorch 如果要 GPU 加速，请按 https://pytorch.org 选择对应 CUDA 版本安装。

## 前置条件

- Python 3.10+
- [wow.export](https://github.com/Kruithne/wow.export) debug build 放在 `C:\wow.export\wow.export\bin\win-x64-debug\`
- 能访问暴雪 CDN（建议选 Taiwan 区域）

## 使用方法

### 启动

双击 `start.vbs`（Windows）或手动运行：

```bash
venv\Scripts\python app.py
```

启动后会弹出加载窗口（约 40 秒加载 CLIP 模型），完成后自动打开浏览器 http://localhost:5001

### 操作流程

1. **Connect** — 选择区域（Taiwan）和版本（Retail），点击 Connect 连接 wow.export 到暴雪 CDN
2. **Extract** — 选择类别（WMO/Creatures/Items 等），点击 Extract All 开始提取特征向量
   - 支持 Resume（同类别断点续传）
   - 不同类别独立存储，互不覆盖
3. **Search** — 上传图片 / 粘贴图片（Ctrl+V）/ 输入文字描述，搜索最相似的 12 个模型

### 关闭

关闭 bat 控制台窗口即可停止所有服务。

## 文件结构

```
wow-asset-finder/
├── app.py                  # Web 后端 (Flask + SocketIO)
├── extract_features.py     # 核心：模型加载、截图、特征提取
├── search.py               # CLI 搜索（可选）
├── templates/
│   └── index.html          # 前端页面
├── splash.ps1              # 启动加载画面
├── start.vbs               # 启动入口（双击运行）
├── requirements.txt        # Python 依赖
├── features/               # 特征向量存储（按类别）
│   ├── features_wmo.npz
│   ├── features_m2_creatures.npz
│   ├── progress_wmo.json
│   └── ...
├── wow_export_data/        # wow.export 的 CASC 缓存（自动生成）
└── venv/                   # Python 虚拟环境
```

## 模型类别

| 选项 | 说明 | 大约数量 |
|------|------|----------|
| All Models | 所有模型 | 143k+ |
| WMO - Buildings & Structures | 城堡、教堂、桥梁、塔楼等建筑 | ~12k |
| Creatures | 生物模型 | - |
| Items | 物品模型 | - |
| Characters | 角色模型 | - |
| Doodads & Props | 桶、栅栏、灯具等小物件 | - |
| Environment | 岩石、树木等环境元素 | - |
| Spells & Effects | 法术特效 | - |

## 注意事项

- wow.export 一次只能运行一个实例
- 首次连接某个区域/版本需要从 CDN 下载数据（较慢），后续有缓存
- 提取所有 WMO 模型约需 8-17 小时，支持断点续传
- 搜索不需要连接 wow.export，只需要有已提取的特征文件
- 缩略图预览需要连接 wow.export（实时渲染，不存储）
