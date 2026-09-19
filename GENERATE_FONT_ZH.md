# 从参考图生成 TTF 字体

`generate_font.py` 使用 `train.py` 训练的生成器和 FontStyleViT，批量读取文件夹中的风格参考图。每张参考图都会生成 52 个大小写字母并导出一个 TTF，最后通过 Pillow/FreeType **重新加载生成的 TTF 文件**渲染英文，并把参考图与字体预览左右拼接。

## 使用方法

在原有训练环境中安装新增的轮廓提取依赖（其余依赖见 `requirements.txt`）：

```bash
python -m pip install opencv-python-headless fonttools Pillow
```

在项目根目录执行，将示例路径替换为实际文件。以下为一行命令，Windows PowerShell 和 Linux 均可使用：

```bash
python generate_font.py --input references --checkpoint outputs/font_cgan_parseq/checkpoints/latest.pt --output outputs/generated_fonts --family-name "My Mimic Font"
```

普通训练 checkpoint 内含训练配置，默认优先使用 `generator_ema`，没有 EMA 时使用 `generator`。也可以通过 `--generator-state generator` 明确选择非 EMA 权重。

如果 FontStyleViT 权重移动了位置，追加 `--style-checkpoint 实际路径.pt`。它必须与生成器训练时使用的风格编码器相匹配。

使用训练结束导出的 `font_generator.pt` 时，传入该次训练的配置文件，以还原参考图预处理和字形画布设置。该导出文件的 `generator` 字段本身已经是 EMA 权重：

```bash
python generate_font.py --input references --checkpoint outputs/font_cgan_parseq/font_generator.pt --config outputs/font_cgan_parseq/resolved_config.yaml --style-checkpoint outputs/font_style_vit_5/checkpoints/epoch_0080.pt --output outputs/generated_fonts
```

参考图应尽量接近训练输入：浅色背景上的深色文字，避免复杂背景。透明区域自动合成白色背景，图像会按训练配置转灰度、缩放、对齐 patch 并归一化。默认使用 CUDA；没有 CUDA 时自动回退 CPU，也可显式指定 `--device cpu`。

## 输出文件

以上命令产生：

```text
outputs/generated_fonts/
  reference_1.ttf
  reference_1_assets/
    alphabet_grid.png
    font_preview.png
    specimen.png
    specimen.txt
    metadata.json
    glyphs/
      U0061.png ... U007A.png   # a-z
      U0041.png ... U005A.png   # A-Z
```

- `reference_1.ttf`：每张输入图对应一个包含 52 个字母、空格及 `.notdef` 的轮廓字体，可由支持 TTF 的软件加载，无需为预览安装到系统。
- `alphabet_grid.png`：模型生成的字形，顺序为 a-z、A-Z；位置预测模型会先将字形裁剪图恢复到原画布。
- `glyphs/`：52 张灰度图。使用 Unicode 编码文件名，避免 Windows 上 `a.png` 和 `A.png` 相互覆盖。
- `font_preview.png`：仅包含使用导出字体文件排版的英文。
- `specimen.png`：左侧为对应的输入风格图，右侧为 `font_preview.png`，便于直接比较风格。
- `metadata.json`：输入和权重路径、推理参数、预测位置（如有）、估计基线、缩放比例及各字母的度量信息。

默认递归处理输入文件夹并在输出目录中保留子目录结构；可用 `--no-recursive` 只处理输入目录第一层。某张图处理失败时会继续处理其余图片，并将结果写入输出根目录的 `batch_summary.json`；使用 `--strict-inputs` 可在第一次失败时立即停止。

默认预览包含大小写字母表，以及全小写和全大写两种形式的英文全字母句：

```text
abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
the quick brown fox jumps over the lazy dog
THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
```

因此默认预览覆盖全部 52 个大小写字母。可用 `--text "The quick brown fox jumps over the lazy dog"` 自定义文本。字体仅生成字母和空格，自定义预览中使用数字、标点等未支持字符会明确报错。

## 轮廓和排版调整

模型生成的是栅格图。脚本通过 [OpenCV 轮廓提取](https://docs.opencv.org/4.10.0/d3/dc0/group__imgproc__shape.html)保留外轮廓、内孔和独立笔画，再用 [fontTools](https://fonttools.readthedocs.io/en/latest/) 写入 TrueType 轮廓及字体表。轮廓为多边形近似。

所有字母共享同一个纵向坐标系和缩放比例，避免把小写字母强行拉高到大写高度。默认根据无下伸部的小写字母估计基线，将估计大写高度映射到 700 字体单位（每 em 为 1000）。水平方向移除图像居中留白，以实际轮廓宽度加两侧留白设置字宽。模型没有预测真实字体的 advance width、kerning 或 hinting，因此这些排版度量是近似值，不会完整恢复原字体的专业排版参数。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--threshold` | `160` | 0–255 灰度中，小于该值的像素作为笔画；提高可保留更浅的笔画，也可能引入噪点 |
| `--min-component-area` | `3` | 删除面积更小的孤立噪点；小字号点状笔画被误删时可设为 `1` |
| `--simplify` | `0.35` | 轮廓简化容差，单位为原图像素；降低可保留更多细节，可设为 `0` |
| `--baseline` | 自动估计 | 基线在原画布高度中的比例，范围 0–1，例如 `0.75` |
| `--side-bearing` | `50` | 每个字母左右两侧的留白，单位为字体单位 |
| `--space-width` | `300` | 空格宽度，单位为字体单位 |
| `--font-size` | `64` | 英文预览字号，单位为像素 |
| `--batch-size` | `13` | 每次推理生成的字母数，显存不足时降低 |
| `--seed` | `42` | 生成器噪声种子 |
| `--no-amp` | 未启用 | 关闭 CUDA 混合精度；出现非有限输出时可尝试 |

字母为空或阈值过滤后没有可用轮廓时，脚本会标明失败字母并报错。已经保存的字形图可用于排查模型输出和阈值。
