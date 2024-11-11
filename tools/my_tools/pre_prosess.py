from PIL import Image
import cv2
import numpy as np

import os
import pathlib


def classify_gt(fp, dest, mode="L"):
    assert os.path.exists(fp), f"{fp}文件不存在"
    image = Image.open(fp)
    image = image.convert(mode)
    # 批量修改图像的所有像素值
    for i in range(image.width):
        for j in range(image.height):
            # 获取当前像素的RGB值
            gray_value = image.getpixel((i, j))
            final_value = 0
            if gray_value > 128:
                final_value = 255
            image.putpixel((i, j), final_value)
    image.save(dest)
    image.close()


def batch_classify_gt(root_dir, dest_dir, mode="L"):
    assert os.path.exists(root_dir), f"{root_dir}目录不存在"
    # 生成目标目录
    import shutil
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.mkdir(dest_dir)
    file_list = os.listdir(root_dir)
    file_list = [file for file in file_list if file.endswith("png")]
    for i in file_list:
        source = os.path.join(root_dir, i)
        dest = os.path.join(dest_dir, i)
        classify_gt(source, dest)


# if __name__ == "__main__":
#     batch_classify_gt("tmp", "tmp/optimized_gt")
