# 本地视觉打标 → Billfish

用本机视觉模型给图片打中文标签，再写进 [Billfish](https://www.billfish.cn/) 素材库。

- **不改原图**，推理时只在内存里缩放。
- 标签来自可编辑的 Billfish 词表，每张图 **1～4 个** 叶子标签。
- 打标适合 **Apple Silicon + MLX**；把结果导入 Billfish 只需 Python 标准库，可在 **Windows** 上跑。

## 流程

```
Mac 上打标  →  results.jsonl  →  Windows 上写入 billfish.db
```

1. 关掉 Chrome 等占内存的程序（16GB 机器建议如此）。
2. 启动网页，填写素材目录，按需改词表，点「开始批量打标」。
3. 把 `results.jsonl`（以及三个 py：`import_billfish.py`、`billfish_vocab.py`、`vocab_store.py`）拷到装了 Billfish 的电脑。
4. **彻底退出 Billfish**，预览匹配数量，再写入。脚本会先备份 `billfish.db`。

## 环境

### Apple Silicon（打标 + 网页）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-apple.txt
```

国内下载模型可加镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

默认模型写在 `tag_images.py` 的 `PRIMARY_MODEL`（MLX 量化视觉模型，约 5GB）。

启动网页：

```bash
python chat_app.py
```

浏览器打开 http://127.0.0.1:8765

- **聊天 / 单张打标**：拖一张图问答或打标签。
- **批量打标**：填本机文件夹、输出目录、词表，看右侧运行日志。
- **导入 Billfish**：填 `素材库\.bf\billfish.db`，先「预览导入」再「写入库」。

也可以不用网页，直接命令行打标：

```bash
python tag_images.py --folder "/path/to/photos" --out outputs/web_batch --resume
```

### Windows（只导入 Billfish）

不需要 MLX。把下面文件放到同一目录：

- `import_billfish.py`
- `billfish_vocab.py`
- `vocab_store.py`
- `results.jsonl`（Mac 上打出来的结果）
- 可选：`config/vocab.json`（如果你在网页里改过词表）

```bat
python import_billfish.py --jsonl results.jsonl --csv billfish_tags.csv --db "D:\图库\.bf\billfish.db"
```

确认匹配数量后真正写入：

```bat
python import_billfish.py --jsonl results.jsonl --db "D:\图库\.bf\billfish.db" --create-missing --replace --apply
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--apply` | 真正写入；不加则只预览 |
| `--create-missing` | 库里没有的分类/叶子就新建 |
| `--replace` | 先去掉这批文件上已有的词表标签再写 |
| `--strip-old` | 剥掉旧的「媒介 / 分级」那套树 |

按文件名匹配（忽略 `.lnk`）。重名文件会全部挂上同一组标签。

## 词表

默认公开词表是摄影向：拍摄对象、拍摄环境、光线、构图、色彩、表现手法、器材技法。分类名只做挂靠，不会当成图片标签。`跳过` 表示疑似未成年，导入时不会写入。

本机若要沿用另一套私人词表，把树存在 `config/vocab.json`（已加入 `.gitignore`，不会上传）。网页里「保存词表」也是写这个文件。

## 安全

- 关闭 Billfish 再写库。写入前会生成 `billfish.db.bak-时间戳`。
- 不要把 `outputs/`、`config/settings.json`、原图库提交到 Git。
- 本工具直接改 SQLite，不是官方导入接口；升级 Billfish 后若表结构变了，先 `--apply` 前看预览，必要时再备份。

## 维护者

- [WindSpringBird](https://github.com/WindSpringBird)

## 许可

按你自己的仓库许可发布。模型权重遵循其 Hugging Face 页面的许可证。
