import os  
from PIL import Image  

# 定义路径  
base_image_dir = '/mnt/afs/xueyingyi/image_vague/image'  
inpainting_demo_dir = '/mnt/afs/xueyingyi/image_vague/inpainting_demo'  
output_dir = '/mnt/afs/xueyingyi/image_vague/resized'  

# 创建输出目录（如果不存在）  
os.makedirs(output_dir, exist_ok=True)  

# 遍历inpainting_demo目录中的所有文件  
for filename in os.listdir(inpainting_demo_dir):  
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # 根据需要添加其他图片格式  
        # 构造基准图片路径  
        base_image_path = os.path.join(base_image_dir, filename)  
        
        # 构造待调整大小的图片路径  
        inpainting_image_path = os.path.join(inpainting_demo_dir, filename)  
        
        # 检查基准图片是否存在  
        if os.path.exists(base_image_path):  
            # 打开基准图片并获取尺寸  
            with Image.open(base_image_path) as base_image:  
                width, height = base_image.size  
                
                # 打开待调整大小的图片  
                with Image.open(inpainting_image_path) as inpainting_image:  
                    # 调整大小  
                    resized_image = inpainting_image.resize((width, height))  
                    
                    # 构造输出文件路径并保存  
                    output_image_path = os.path.join(output_dir, filename)  
                    resized_image.save(output_image_path)  

                    print(f'Resized {filename} and saved to {output_dir}')  
        else:  
            print(f'Base image not found for {filename}, skipping...')