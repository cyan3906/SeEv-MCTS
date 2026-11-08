# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import os

# # 配置matplotlib后端
# # 优先尝试使用TkAgg后端（有图形界面的环境）
# try:
#     matplotlib.use('TkAgg')
#     print("已设置TkAgg后端")
# except:
#     # 如果TkAgg不可用，使用Agg后端（非交互式，保存图片）
#     matplotlib.use('Agg')
#     print("已设置Agg后端（非交互式）")

# # 配置路径
# FOLDER_PATH = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/images/testmini/ALL"
# IMAGE_NAME = "10.jpg"
# img_path = os.path.join(FOLDER_PATH, IMAGE_NAME)

# if os.path.exists(img_path):
#     print(f"图片存在: {img_path}")
#     # 读取图片
#     img = mpimg.imread(img_path)
#     print(f"图片形状: {img.shape}")
    
#     # 创建图形
#     plt.figure(figsize=(8, 6))
#     plt.imshow(img)
#     plt.axis("off")
#     plt.title("Visualized Image")
    
#     try:
#         # 尝试显示图片
#         plt.show()
#         print("图片显示成功")
#     except Exception as e:
#         print(f"显示图片时出错: {e}")
#         # 保存图片到当前目录
#         save_path = "displayed_image.jpg"
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#         print(f"图片已保存到: {os.path.abspath(save_path)}")
# else:
#     print(f"❌ 图片不存在：{img_path}")

# 替代方案：使用pillow库显示图片
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 配置matplotlib后端
# 优先尝试使用TkAgg后端（有图形界面的环境）
try:
    matplotlib.use('TkAgg')
    print("已设置TkAgg后端")
except:
    # 如果TkAgg不可用，使用Agg后端（非交互式，保存图片）
    matplotlib.use('Agg')
    print("已设置Agg后端（非交互式）")

# 配置路径
FOLDER_PATH = "/home/jncsnlp4/SSD2/ZJJ/VReST/data/MathVista/data/images/testmini/ALL"
IMAGE_NAME = "10.jpg"
img_path = os.path.join(FOLDER_PATH, IMAGE_NAME)

if os.path.exists(img_path):
    print(f"图片存在: {img_path}")
    # 读取图片
    img = mpimg.imread(img_path)
    print(f"图片形状: {img.shape}")
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Visualized Image")
    
    try:
        # 尝试显示图片
        plt.show()
        print("图片显示成功")
    except Exception as e:
        print(f"显示图片时出错: {e}")
        # 保存图片到当前目录
        save_path = "displayed_image.jpg"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"图片已保存到: {os.path.abspath(save_path)}")
else:
    print(f"❌ 图片不存在：{img_path}")
    
# MWP:
# 45
# 71
# 72
# 205
# 258
# 331
# 332
# 359
# 517
# 524
# 867
# 978

# FQA:
# 38
# 108
# 127
# 133
# 136
# 188
# 202
# 209
# 213
# 216

241 11,13
253 2,3
288 0-0.2
291 D和E
305 9 6
380 4 5
400 4
422 10 1
429 10 1
432 4 1
462 1 3
470 C
496 1 10
509 yes no


file1 = "/home/jncsnlp4/SSD2/ZJJ/VRest-data/data_manager/mathvistra/ours2v2.2/FQA"

print(len(os.listdir(file1)))
