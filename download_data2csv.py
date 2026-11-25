# 下载 COCO 2014 Caption 数据（脚本免加载版）
# 原仓库依赖 modelscope 的 coco_2014_caption 脚本加载，但当前环境的
# datasets 5.x 已彻底移除“脚本式数据集”支持，故改为直接从官方 COCO 源
# 下载标注压缩包 + 图片，输出格式与原来一致（image_path / caption），
# 以便 csv2json.py 无需任何改动即可继续生成 data_vl.json。
import os
import io
import json
import time
import zipfile
import urllib.request
import pandas as pd

MAX_DATA_NUMBER = 500
ANNOT_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
IMG_BASE = "http://images.cocodataset.org/train2014/"
OUT_DIR = "coco_2014_caption"
CSV_PATH = "./coco-2024-dataset.csv"
ZIP_TMP = "annotations_trainval2014.zip"


def download_to_file(url, dest, max_retries=6, timeout=180, chunk=1 << 20):
    """带重试的下载（流式写盘），失败自动退避重试。"""
    last = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (Qwen3-VL-MoeLORA setup)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
            return dest
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  [retry {attempt}/{max_retries}] {url} -> {e}")
            time.sleep(3 * attempt)
    raise last


if __name__ == "__main__":
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        print("coco-2024-dataset.csv 已存在, 跳过数据处理步骤")
    else:
        # 1) 下载标注压缩包并解压出 captions_train2014.json
        print("下载 COCO 标注压缩包 ...")
        download_to_file(ANNOT_URL, ZIP_TMP)
        with zipfile.ZipFile(ZIP_TMP) as z:
            cname = [n for n in z.namelist() if n.endswith("captions_train2014.json")][0]
            captions = json.loads(z.read(cname))
        os.remove(ZIP_TMP)

        # 2) 建立 image_id -> file_name 映射，并取前 MAX_DATA_NUMBER 条标注
        images = {img["id"]: img["file_name"] for img in captions["images"]}
        annotations = captions["annotations"]
        selected = annotations[:MAX_DATA_NUMBER]

        # 3) 下载图片（同图去重，仅下载一次）并记录绝对路径 + 描述
        os.makedirs(OUT_DIR, exist_ok=True)
        image_paths = []
        captions_list = []
        done = set()
        for i, ann in enumerate(selected):
            image_id = ann["image_id"]
            caption = str(ann["caption"]).strip()
            img_path = os.path.abspath(os.path.join(OUT_DIR, f"{image_id}.jpg"))
            if image_id not in done:
                download_to_file(IMG_BASE + images[image_id], img_path)
                done.add(image_id)
            image_paths.append(img_path)
            captions_list.append(caption)
            if (i + 1) % 50 == 0:
                print(f"Processing {i + 1}/{len(selected)} images ({(i + 1) / len(selected) * 100:.1f}%)")

        # 4) 落盘 CSV（列名与原来一致：image_path / caption）
        df = pd.DataFrame({"image_path": image_paths, "caption": captions_list})
        df.to_csv(CSV_PATH, index=False)
        print(f"数据处理完成，共处理了{len(selected)}张图片描述")
