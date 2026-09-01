# SD与MJ魔法词

# 一、Midjourney 提示词

**中文版（画面构思参考）：**

> 午后，一位气质温婉的中国美女坐在书房窗边静静读书，金色阳光透过木质窗棂洒落，光束中漂浮着细小尘埃，背景是摆满书籍的高大木质书架，桌上放着一杯清茶，氛围宁静温馨，电影感光影，柔和阴影，浅景深，写实风格，超高细节，8k画质

**英文版（可直接使用）：**

```
/imagine prompt: a beautiful elegant Chinese young woman reading a book in a cozy study room in the afternoon, warm golden sunlight streaming through the wooden lattice window, floating dust particles in the light beams, tall wooden bookshelves filled with books in the background, a cup of tea on the desk, peaceful and serene atmosphere, cinematic lighting, soft shadows, shallow depth of field, photorealistic, ultra detailed, 8k --ar 3:4 --v 6.1 --style raw
```

**参数说明：**
- `--ar 3:4`：竖构图，适合人物场景（可改为 16:9 横构图）
- `--v 6.1`：模型版本（按你实际版本调整）
- `--style raw`：减少 MJ 默认美化，更写实；若想要插画感可删除此参数

---

# 二、Stable Diffusion 提示词

**正向提示词：**

```
masterpiece, best quality, ultra detailed, 8k, photorealistic, RAW photo, 1girl, solo, beautiful Chinese young woman, elegant, gentle expression, long black hair, white blouse, reading a book, holding book, sitting by window, cozy study room, wooden bookshelves, books stacked on desk, cup of tea, afternoon, golden sunlight, sunbeam through window, light rays, dust particles, warm lighting, soft shadows, depth of field, blurry background, detailed face, detailed eyes, detailed skin, cinematic composition, upper body
```

**负向提示词：**

```
lowres, worst quality, low quality, jpeg artifacts, blurry, bad anatomy, bad hands, bad proportions, missing fingers, extra fingers, extra digits, fewer digits, fused fingers, extra limbs, deformed, disfigured, mutated hands, long neck, bad eyes, cross-eyed, asymmetric eyes, ugly, duplicate, cropped, out of frame, watermark, signature, username, text, logo, error, nsfw
```

**推荐参数：**

| 项目 | 建议值 |
|---|---|
| 采样器 | DPM++ 2M Karras |
| 步数 | 25–30 |
| CFG Scale | 6–7 |
| 分辨率 | 768×1024 或 832×1216（竖图） |
| 高清修复 | 开启，放大 1.5 倍，重绘幅度 0.3–0.4 |

**小贴士：**
- 若使用 WebUI，可对重点词加权，如 `(masterpiece:1.2), (best quality:1.2)`，权重不必超过 1.4，否则画面易崩。
- 想要国风氛围，可在正向加入 `hanfu, chinese architecture`（汉服/中式建筑）；想换插画风格，把 `photorealistic, RAW photo` 替换为 `illustration, chinese art style`。
- Midjourney 无需负向提示词，它会自动规避劣化；SD 则强烈建议保留负向词以保证手部和面部质量。
