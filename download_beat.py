"""
BEAT dataset downloader via hf-mirror.com
绕过 huggingface_hub 新版 list_repo_tree 分页 cursor bug。
直接用 requests 下载，支持断点续传、自动重试。

用法：
    python download_beat.py
"""

import os
import time
import requests
from pathlib import Path

# ===== 配置 =====
MIRROR        = "https://hf-mirror.com"
REPO_ID       = "H-Liu1997/BEAT"
LOCAL_DIR     = r"f:\study\DiffSHEG\data\BEAT\raw"
# 只下载 4 位英文说话人（说话人编号1-4）
# 实际路径格式: beat_english_v0.2.1/beat_english_v0.2.1/<id>/
INCLUDE_PREFIXES = [
    "beat_english_v0.2.1/beat_english_v0.2.1/1/",   # wayne  (id=1)
    "beat_english_v0.2.1/beat_english_v0.2.1/2/",   # scott  (id=2)
    "beat_english_v0.2.1/beat_english_v0.2.1/3/",   # chemistry (id=3)
    "beat_english_v0.2.1/beat_english_v0.2.1/4/",   # carlos (id=4)
]
# 只下载训练需要的格式：
#   .bvh  → 骨骼旋转姿态
#   .wav  → 原始音频（后续转 16kHz .npy）
#   .json → 面部表情 BlendShape
#   .txt  → 语义得分标注（sem_rep，用于 loss 加权）
# 排除 .TextGrid（音素对齐，代码未使用）和 .csv（BVH 的冗余备份）

INCLUDE_EXTENSIONS = {".bvh", ".wav", ".json", ".txt"}

MAX_RETRIES   = 10
RETRY_WAIT    = 5   # seconds between retries
# =================

API_BASE  = f"{MIRROR}/api/datasets/{REPO_ID}"
FILE_BASE = f"{MIRROR}/datasets/{REPO_ID}/resolve/main"

session = requests.Session()
session.headers.update({"User-Agent": "python-requests/beat-downloader"})


def list_all_files():
    """列出仓库中的所有文件（手动处理分页，将 cursor URL 中的域名替换为镜像）。"""
    files = []
    url = f"{API_BASE}/tree/main"
    params = {"recursive": "true", "expand": "false", "limit": 1000}

    while url:
        for attempt in range(MAX_RETRIES):
            try:
                r = session.get(url, params=params, timeout=60)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                print(f"  [列举文件] 失败 (尝试 {attempt+1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_WAIT)
        else:
            print("  列举文件多次失败，停止。")
            return files

        # data 可能是 list（每个 item 有字段 path/type/size），
        # 也可能是 dict（带 nextHref 或 Link header）
        items = data if isinstance(data, list) else data.get("items", data)
        for item in items:
            if isinstance(item, dict) and item.get("type") == "file":
                files.append(item["path"])

        # 处理下一页：Link header 或 nextHref 字段，将域名替换为镜像
        next_url = None
        link_header = r.headers.get("Link", "")
        if link_header:
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    raw_next = part.split(";")[0].strip().strip("<>")
                    # 替换域名为镜像
                    next_url = raw_next.replace("https://huggingface.co", MIRROR)
                    break
        if not next_url and isinstance(data, dict):
            href = data.get("nextHref") or data.get("next")
            if href:
                next_url = href.replace("https://huggingface.co", MIRROR)

        url = next_url
        params = {}  # 后续分页 URL 里已经包含了参数

    return files


def download_file(remote_path: str):
    """下载单个文件，支持断点续传和重试。"""
    # 前缀过滤
    if INCLUDE_PREFIXES:
        if not any(remote_path.startswith(p) for p in INCLUDE_PREFIXES):
            return
    # 扩展名过滤
    if INCLUDE_EXTENSIONS:
        ext = "." + remote_path.rsplit(".", 1)[-1] if "." in remote_path else ""
        if ext not in INCLUDE_EXTENSIONS:
            return

    local_path = Path(LOCAL_DIR) / remote_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"{FILE_BASE}/{remote_path}"

    for attempt in range(MAX_RETRIES):
        # 支持断点续传
        headers = {}
        existing_size = local_path.stat().st_size if local_path.exists() else 0

        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        try:
            r = session.get(url, headers=headers, stream=True, timeout=60)
            if r.status_code == 416:
                # Range not satisfiable → 文件已完整
                print(f"  ✓ 已完整: {remote_path}")
                return
            r.raise_for_status()

            mode = "ab" if existing_size > 0 and r.status_code == 206 else "wb"
            total = int(r.headers.get("Content-Length", 0))
            downloaded = existing_size if mode == "ab" else 0

            with open(local_path, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)

            print(f"  ✓ {remote_path}  ({downloaded/1024/1024:.1f} MB)")
            return

        except Exception as e:
            print(f"  [下载] {remote_path} 失败 (尝试 {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_WAIT)

    print(f"  ✗ 放弃: {remote_path}")


def main():
    print(f"=== BEAT 数据集下载器 ===")
    print(f"镜像: {MIRROR}")
    print(f"输出目录: {LOCAL_DIR}")
    print(f"过滤前缀: {INCLUDE_PREFIXES or '全部'}")
    print()

    # 1. 列出文件
    print("正在列举仓库文件...")
    all_files = list_all_files()
    print(f"共找到 {len(all_files)} 个文件")
    print("\n前 30 个文件路径（用于确认前缀）：")
    for p in all_files[:30]:
        print(f"  {p}")

    # 2. 过滤（前缀 + 扩展名）
    def should_download(p):
        if INCLUDE_PREFIXES and not any(p.startswith(x) for x in INCLUDE_PREFIXES):
            return False
        if INCLUDE_EXTENSIONS:
            ext = "." + p.rsplit(".", 1)[-1] if "." in p else ""
            if ext not in INCLUDE_EXTENSIONS:
                return False
        return True

    filtered = [f for f in all_files if should_download(f)]
    print(f"\n需要下载: {len(filtered)} 个文件\n")

    # 3. 逐文件下载
    for i, fpath in enumerate(filtered, 1):
        print(f"[{i}/{len(filtered)}] {fpath}")
        download_file(fpath)

    print("\n=== 下载完成 ===")


if __name__ == "__main__":
    main()
