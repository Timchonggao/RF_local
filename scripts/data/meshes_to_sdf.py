import os
import trimesh
import mesh2sdf
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 设置路径
base_path_template = '/data3/gaochong/project/RadianceFieldStudio/data/multiview_dynamic_blender/{}/obj/'
output_path_template = '/data3/gaochong/project/RadianceFieldStudio/data/multiview_dynamic_blender/{}/mesh_gt_preprocess/'

mesh_scale = 0.8
size = 128
level = 2 / size

categories = ['toy']

def process_category(category):
    base_path = base_path_template.format(category)
    output_path = output_path_template.format(category)
    os.makedirs(output_path, exist_ok=True)
    
    # 获取该类别下的所有 .obj 文件，并按文件名排序
    obj_files = [f for f in os.listdir(base_path) if f.endswith('.obj')]
    obj_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))  # 按数字排序文件名
    
    # 循环处理该类别下的每个网格文件
    for filename in tqdm(obj_files, desc=f'Processing {category}', unit='frame'):
        file_path = os.path.join(base_path, filename)
        
        # 加载网格
        mesh = trimesh.load(file_path, force='mesh')

        # 归一化网格
        vertices = mesh.vertices
        vertices = vertices * 1.6 # 对原始的toy数据，mesh都在+/- 0.5的范围，所以，统一缩放到 +/- 0.8的范围

        # bbmin = vertices.min(0)
        # bbmax = vertices.max(0)
        # center = (bbmin + bbmax) * 0.5
        # scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
        # vertices = (vertices - center) * scale

        # 计算SDF并修复网格
        t0 = time.time()
        sdf, mesh = mesh2sdf.compute(
            vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
        t1 = time.time()

        # 恢复网格并输出
        # mesh.vertices = mesh.vertices / scale + center
        output_filename = os.path.join(output_path, filename.replace('.obj', '.fixed.obj'))
        mesh.export(output_filename)
        np.save(output_filename.replace('.obj', '.npy'), sdf)
        
        # 打印每一帧的处理时间
        print(f'{category} - {filename} processed in {t1 - t0:.4f} seconds')

def main():
    # # 创建进程池，并行处理不同类别
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(process_category, category) for category in categories]
        
    #     # 显示进度条并等待所有类别处理完成
    #     for future in as_completed(futures):
    #         pass  # 可以在这里添加更多的处理或输出

    # print('All categories processed.')
    process_category('toy')

if __name__ == '__main__':
    main()
