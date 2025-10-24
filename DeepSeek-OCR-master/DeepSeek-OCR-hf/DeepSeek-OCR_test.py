"""
DeepSeek-OCR 测试脚本
"""
from transformers import AutoModel, AutoTokenizer
import torch
import os
import time
import psutil
import GPUtil
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
# ============ 配置区 ============
# 1. 模型路径
model_name = 'E:/Ollama/DeepSeek-OCR_models'  # 从 HuggingFace 下载
# 2. 设置图片/PDF路径
IMAGE_PATH = 'E:/Ollama/DeepSeek-OCR/assets/show2.jpg'
# 3. 设置输出目录
BASE_OUTPUT_DIR = 'E:/Ollama/output'
# 根据图片文件名生成新的输出目录
image_name = Path(IMAGE_PATH).stem
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, image_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 4. 设置任务类型：'markdown' 或 'ocr'
TASK = 'ocr'
# 5. GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# =================================
def main():
    print("=" * 70)
    print("加载模型...")
    # 记录初始状态
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    # 加载模型
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True  # 只使用本地缓存，不联网检查更新
    )
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True  # 只使用本地缓存，不联网检查更新
    )
    model = model.eval().cuda().to(torch.bfloat16)
    load_time = time.time() - load_start
    # 记录加载后状态
    after_load_memory = process.memory_info().rss / 1024 / 1024  # MB
    gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
    print(f"✓ 模型加载完成 (耗时: {load_time:.2f}秒)")
    print(f"内存占用: {after_load_memory - initial_memory:.2f} MB")
    if gpu:
        print(f"显存占用: {gpu.memoryUsed:.2f} MB / {gpu.memoryTotal:.2f} MB")
    print("=" * 70)
    # 设置提示词
    if TASK == 'markdown':
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    else:
        prompt = "<image>\nFree OCR. " # 移除 <|grounding|>
    # 检查文件类型并转换PDF
    file_path = Path(IMAGE_PATH)
    if file_path.suffix.lower() == '.pdf':
        print(f"\n检测到PDF文件: {IMAGE_PATH}")
        print("正在转换PDF为图片...")
        # 打开PDF
        pdf_doc = fitz.open(IMAGE_PATH)
        total_pages = len(pdf_doc)
        print(f"PDF共 {total_pages} 页")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        all_results = []
        total_infer_time = 0
        # 逐页处理
        for page_num in range(total_pages):
            print(f"\n{'='*70}")
            print(f"处理第 {page_num + 1}/{total_pages} 页")
            print("-" * 70)
            # 转换当前页为图片
            page = pdf_doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            # 保存临时图片
            temp_image_path = f'{OUTPUT_DIR}/temp_page_{page_num + 1}.png'
            pix.save(temp_image_path)
            # 记录推理前状态
            infer_start = time.time()
            cpu_percent_start = psutil.cpu_percent(interval=0.1)
            gpu_util_start = gpu.load * 100 if gpu else 0
            # 执行OCR
            page_output_dir = f'{OUTPUT_DIR}/page_{page_num + 1}'
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=page_output_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True
            )
            # 记录推理后状态
            infer_time = time.time() - infer_start
            total_infer_time += infer_time
            # 读取保存的结果文件
            result_file = f'{page_output_dir}/result.mmd'
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    page_result = f.read()
                # 复制该页的images目录到output/images下
                page_images_dir = f'{page_output_dir}/images'
                if os.path.exists(page_images_dir):
                    output_images_dir = f'{OUTPUT_DIR}/images'
                    os.makedirs(output_images_dir, exist_ok=True)
                    # 复制图片并重命名（添加页码前缀）
                    import shutil
                    for img_file in os.listdir(page_images_dir):
                        src = os.path.join(page_images_dir, img_file)
                        dst = os.path.join(output_images_dir, f'page{page_num + 1}_{img_file}')
                        shutil.copy2(src, dst)
                    # 更新结果中的图片路径
                    page_result = page_result.replace('](images/', f'](images/page{page_num + 1}_')
                all_results.append(f"\n\n# 第 {page_num + 1} 页\n\n{page_result}")
                print(f"✓ 第 {page_num + 1} 页识别完成 (耗时: {infer_time:.2f}秒)")
            else:
                print(f"✗ 第 {page_num + 1} 页识别失败")
        pdf_doc.close()
        # 合并所有页结果
        result = "\n".join(all_results)
        # 保存完整结果
        with open(f'{OUTPUT_DIR}/full_result.md', 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n{'='*70}")
        print("PDF处理完成")
        print("=" * 70)
        print(f"总页数:       {total_pages}")
        print(f"总耗时:       {total_infer_time:.2f} 秒")
        print(f"平均每页:     {total_infer_time/total_pages:.2f} 秒")
        # 获取最终状态
        final_memory = process.memory_info().rss / 1024 / 1024
        cpu_percent_end = psutil.cpu_percent(interval=0.1)
        if gpu:
            gpu = GPUtil.getGPUs()[0]
            gpu_util_end = gpu.load * 100
        else:
            gpu_util_end = 0
        print(f"内存使用:     {final_memory:.2f} MB")
        if gpu:
            print(f"显存占用:     {gpu.memoryUsed:.2f} MB / {gpu.memoryTotal:.2f} MB")
        print("=" * 70)
    else:
        # 处理单个图片
        process_path = IMAGE_PATH
        print(f"\n处理文件: {IMAGE_PATH}")
        print("-" * 70)
        # 记录推理前状态
        infer_start = time.time()
        cpu_percent_start = psutil.cpu_percent(interval=0.1)
        gpu_util_start = gpu.load * 100 if gpu else 0
        infer_result = model.infer( # 将返回值赋给 infer_result
            tokenizer,
            prompt=prompt,
            image_file=process_path,
            output_path=OUTPUT_DIR,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True
        )
        print(f"model.infer 返回值: {infer_result}") # 打印返回值
        # 记录推理后状态
        infer_time = time.time() - infer_start
        cpu_percent_end = psutil.cpu_percent(interval=0.1)
        final_memory = process.memory_info().rss / 1024 / 1024
        if gpu:
            gpu = GPUtil.getGPUs()[0]
            gpu_util_end = gpu.load * 100
        else:
            gpu_util_end = 0

        # 读取保存的结果文件
        result_file = f'{OUTPUT_DIR}/result.mmd'
        print(f"尝试读取结果文件: {result_file}") # 添加这行
        result = ""
        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                result = f.read()
        # 输出性能统计
        print("-" * 70)
        print("\n性能统计:")
        print("=" * 70)
        print(f"推理耗时:     {infer_time:.2f} 秒")
        print(f"CPU 使用率:   {cpu_percent_end:.1f}%")
        print(f"内存使用:     {final_memory:.2f} MB (推理增加: {final_memory - after_load_memory:.2f} MB)")
        if gpu:
            print(f"GPU 使用率:   {gpu_util_end:.1f}%")
            print(f"显存占用:     {gpu.memoryUsed:.2f} MB / {gpu.memoryTotal:.2f} MB ({gpu.memoryUsed/gpu.memoryTotal*100:.1f}%)")
        print("=" * 70)
    # 显示结果预览
    print(f"\n识别结果预览:")
    print("-" * 70)
    if result:
        preview = result[:300] if len(result) > 300 else result
        print(preview)
        if len(result) > 300:
            print(f"\n... (共 {len(result)} 字符)")
    else:
        print("未获取到结果")
    print("-" * 70)
    print(f"\n✓ 完整结果已保存到: {OUTPUT_DIR}/full_result.md" if file_path.suffix.lower() == '.pdf' else f"\n✓ 完整结果已保存到: {OUTPUT_DIR}")
if __name__ == '__main__':
    main()
