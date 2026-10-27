<p align="center">
  <img src="assets/logo.svg" alt="ygo" width="200" height="40">
</p>

[English](README.md) | **简体中文**

# ygo

### 游戏王卡组与卡图实验场 —— 一份个人实验存档

**在 Python 里直接跑在 `ggml` 上的 DINOv2：手写的 GGUF 加载器与张量图，不用任何框架。算出来的
embedding 全部用来解决 hobby 问题：自动裁切卡图、从一局录像里挑出每个镜头的代表帧、把一堆
`.ydk` 卡组整理并命名成能再次找到的档案。**

这里的每个文件都是为了回答某个关于真实卡、卡组或录像的问题才写的。它需要的数据
（`cards.cdb`、DINOv2 GGUF、`deck/` 语料）**不在**仓库里；这里也没有任何打包、版本或发布。
请把它当作可读的源码，而不是装来即用的工具。

---

## 快速开始

没有包元数据、没有安装目标、没有统一入口。每个文件各自独立，跑得最快的是统计脚本，只需要在
当前目录放一份 `cards.cdb`（YGOPRO 的 sqlite 卡库）：

```bash
git clone https://github.com/ZisIsNotZis/ygo.git
cd ygo
# 把 YGOPRO 的 cards.cdb 放到这里
pip install pyyaml        # stat.py：sqlite3（标准库）+ yaml
python3 stat.py           # 以 YAML 打印种族 / 属性的系列统计
```

各文件所需依赖（从 import 读出来的，没有锁定版本）：

| 文件 | 依赖 |
| --- | --- |
| `stat.py` | Python 3.8+、`pyyaml`、`cards.cdb` |
| `gg.py` | Python **3.12+**（`def dbg[**P, T]`，PEP 695 泛型）、外部 `ggml` Python 模块、`numpy` |
| `dino2.py`、`cut.py`、`cutvid.py` | 以上再加 `pillow`、`scikit-learn`；`cutvid.py` 还需要带 QSV（Intel 硬解）的 `ffmpeg`/`ffprobe` |
| `ydkrename.py` | `xxhash`，以及 `../cards.cdb`、`../strings.conf`、`../pack/` |
| 各 notebook | `numpy`、`scipy`、`torch`（2.x，训练类需要 CUDA）、`cards.cdb`、`lflist.conf`、一份 `deck/**/*.ydk` 语料 |
| `clipdown.sh` / `ypicdown.sh` | `clipnotify`、`xclip`、`sqlite3`、`curl`、ImageMagick（`mogrify`） |

`ggml` 模块是按顶层包 import 的，**并未**随仓库附带；`dino2.init()` 还写死了一个 GGUF 绝对路径
（`/home/z/gguf/dinov2-with-registers-small-imagenet1k-1-layer-f16.gguf`）。在跑 `dino2.py`
以下的任何东西之前，先把它改成你自己导出的 DINOv2 ViT-with-registers。

## 现在能做什么

- **`gg.py` —— 从零写起的 GGML/GGUF 层。** 在 `ggml_tensor_p` 上包了一层 `Tensor`：可读的
  shape/stride、基于 view 的 `flatten`/`permute`/`T`、`cont`、`cat`、与 numpy 互转，以及一个 ViT
  需要的算子：`conv`、`lin`、`mlp`、`swiglu`、`norm`、`softmax`、`gelu`/`silu`、`rope`。此外还有
  构图、`ggml_gallocr` 分配、后端 compute，以及一个会打印每个算子形状的 `@dbg` 装饰器。
- **`dino2.py` —— 纯 ggml 的 DINOv2 推理。** patch embedding 卷积、class 与 register token、
  双三次插值的 position embedding、逐层的 attention/norm/layer-scale，40 层变体走 SwiGLU。返回
  CLS 向量**和** patch 网格；设 `PCA=1` 还会把 patch 特征的 PCA 彩色图写成 JPEG。
- **`cut.py` —— 卡图自动裁切。** 对每张非正方形图做 embedding，把 patch 网格用 PCA 压成一通道，
  沿长边求和、在累加和峰值处下刀 —— 也就是找到画面本体、丢掉边框。
- **`cutvid.py` —— 录像镜头选取。** 用 `ffmpeg`（QSV）以 3 fps 抽出 16×16 灰度帧，丢掉几乎不变
  的连续段，对剩下的做 DINOv2 embedding，每个 k-means 簇留一帧，再按 PCA 投影裁出动作所在的横带
  —— 每个镜头输出一张 `.webp`。环境变量：`X0 Y0 X1 Y1 K0 K1 R ENC VIS`。
- **`ydkrename.py` —— 自己给自己命名的卡组档案。** 单卡最多 3 张、整副最多 75 张，用 `../pack/`
  的卡包列表给卡组定日期，然后贪心地追加卡名、卡描述和系列名里共同出现的最长高频 n-gram，直到
  计数低于 `CNT`；结果形如 `YYMMDD<系列>/<xxhash64>.ydk`。
- **`name.ipynb`** —— 同一命名思路的更早版本，只用 n-gram。
- **`tok.ipynb`** —— 只靠 numpy 的卡名特征选择：两两共现位打包、按熵合并，不用分词库。
- **`model*.ipynb` —— torch 卡组模型。** 在卡 id 序列上的因果 transformer 编码器（`Embedding`/
  `EmbeddingBag`、`SwiGLU`、`MultiheadAttention`、残差 `Sequential`、手写 `triu`/`tril` 下标
  生成）；`deck1inf.ipynb` 从三张种子卡自回归补全整副卡组，同时守住 40/15 的分区和单卡禁限张数。
- **`deck*.ipynb`** —— 数据准备：别名归并、额外卡组类型掩码（`0x4802040`）、按 `lflist.conf`
  取张数上限、卡的共现矩阵。
- **`stat.py`** —— 种族/属性分布与交叉表，输出成 YAML。
- **`cdb2.sql`** —— 从旧的文本导出重建一份兼容 YGOPRO 的 `cards.cdb`（`datas` + `texts`、
  类型/种族/属性位掩码、连接箭头）。
- **`clipdown.sh` / `ypicdown.sh`** —— 从剪贴板抓 `.ydk` 文本存成时间戳文件，以及把数据库里有、
  `pics/` 里缺的卡图从 CDN 补齐。

## 架构

两条互不依赖的流水线，共用同一个 sqlite 库和同一个 embedding 模型：

```
cards.cdb (YGOPRO sqlite)          DINOv2 f16 GGUF
        |                                  |
        |                    gg.py  (ctypes -> libggml/gguf，构图 + gallocr)
        |                                  |
        |                           dino2.py (CLS + patch 网格)
        |                             /          \
  stat.py、name/                cut.py        cutvid.py
  tok/deck*/model*            (卡图)          (视频帧)
  notebook                        |               |
        |                    裁好的图       各镜头的 .webp
  ydkrename.py -> YYMMDD<系列>/<hash>.ydk
```

notebook 之间没有 import 关系，它们只是实验快照，每个都去加载上一步存下的 `cards.*` / `deck*`
pickle。唯一的共享代码是 `gg.py`：`dino2.py` 用它建模，`cutvid.py` 从这里 import。

```
ygo/
├── gg.py            # ggml/gguf ctypes 层：Tensor、算子、图、分配器
├── dino2.py         # ggml 上的 DINOv2 ViT 前向；写出 *.dino2.*f 张量
├── cut.py           # 卡图的 PCA 自动裁切
├── cutvid.py        # 流 -> 去重 -> embedding -> k-means -> 每个镜头裁一张
├── ydkrename.py     # n-gram 卡组命名与归档
├── stat.py          # 种族/属性统计，输出 YAML
├── cdb2.sql         # 从文本导出重建 cards.cdb
├── clipdown.sh
├── ypicdown.sh
├── deck.7z          # 压缩的卡组语料（notebook 期望解压成 deck/）
├── deck*.ipynb      # 数据准备与最早的卡组模型
├── model*.ipynb     # transformer 卡组模型；*inf 只做推理
├── name.ipynb、tok.ipynb、nomodel.ipynb
├── docs/project-status.md
└── LICENSE
```

## 如何开发

没有构建、没有测试套件、没有 linter 配置、没有 CI —— 这是有意的：它是存档，能跑的只有你眼前
看到的东西。唯一存在的自动检查是语法：

```bash
python3 -m py_compile gg.py dino2.py cut.py cutvid.py ydkrename.py stat.py
```

验证只能手做：拿你自己的 `cards.cdb`、GGUF 和图片跑某个文件，然后看输出（`stat.py` 打印
YAML；`cut.py` **原地**改写图片，请先复制；`PCA=1` 会让 `dino2.py` 额外输出可视化）。

这里的 Python 用上了 PEP 695 泛型和大量表达式内解包（`gg.py`、`ydkrename.py`、`cutvid.py`），
所以这些文件需要 3.12+；依赖就是 import 里写的那些，既不锁定也不跟踪。

## 如何贡献

这是一份个人存档 —— 见 [CONTRIBUTING.md](CONTRIBUTING.md)。能复现的真实 bug 欢迎报 issue；
功能需求不欢迎，因为这里不做维护。

## 路线图

**已实现**

- Python 里的 GGML/GGUF 张量层与 DINOv2 推理（`gg.py`、`dino2.py`）
- 基于 DINOv2 embedding 的卡图裁切与视频镜头选取（`cut.py`、`cutvid.py`）
- 用卡面文本 n-gram 给卡组命名归档（`ydkrename.py`、`name.ipynb`）
- 卡组数据准备、统计与 `cards.cdb` 重建（`deck*.ipynb`、`stat.py`、`cdb2.sql`）
- transformer 卡组模型与自回归补全卡组（`model*.ipynb`、`deck1inf.ipynb`）

**计划：没有。** 这个仓库是一堆做完的、不再维护的实验，留着只因为当初的问题和答案都还有意思。
如果哪天真要接着做，最明显的下一步是把 `dino2.init()` 里写死的 GGUF 路径拆掉，以及给
`deck1inf.ipynb` 换上真正的卡组合法性检查、而不是现在这个主/额外计数器 —— 但这只是设想，不是承诺。

## 许可

[AGPL-3.0](LICENSE) —— 见 `LICENSE`。它依赖的外部数据（`cards.cdb`、第三方 CDN 的卡图、
DINOv2 权重）各有自己的许可，不在此许可覆盖范围内。
