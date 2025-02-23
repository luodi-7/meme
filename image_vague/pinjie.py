import cv2
import os

# 定义三个路径
path1 = '/mnt/afs/xueyingyi/image_vague/image'
path2 = '/mnt/afs/xueyingyi/image_vague/inpainting_demo'
path3 = '/mnt/afs/xueyingyi/image_vague/mask_demo'

# 获取第一个路径下的所有图片文件名
image_files = os.listdir(path1)

# 遍历每个图片文件
for image_file in image_files:
    # 读取三个路径下的同名图片
    img1 = cv2.imread(os.path.join(path1, image_file))
    img2 = cv2.imread(os.path.join(path2, image_file))
    img3 = cv2.imread(os.path.join(path3, image_file))

    # 如果图片读取成功
    if img1 is not None and img2 is not None and img3 is not None:
        # 获取第一个图片的尺寸
        height, width = img1.shape[:2]

        # 将其他图片缩放到第一个图片的尺寸
        img2_resized = cv2.resize(img2, (width, height))
        img3_resized = cv2.resize(img3, (width, height))

        # 横向拼接图片
        combined_image = cv2.hconcat([img1, img2_resized, img3_resized])

        # 保存拼接后的图片
        output_path = os.path.join('/mnt/afs/xueyingyi/image_vague/combined', image_file)
        cv2.imwrite(output_path, combined_image)

        print(f"Saved combined image: {output_path}")
    else:
        print(f"Failed to read one or more images for: {image_file}")