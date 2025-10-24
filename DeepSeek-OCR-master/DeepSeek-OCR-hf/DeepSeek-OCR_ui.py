"""
DeepSeek-OCR PyQt6 UI版本
作者: tiandao
"""
import os
import sys
import time
import shutil
from pathlib import Path

import fitz  # PyMuPDF
import torch
from transformers import AutoModel, AutoTokenizer

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QLabel, QComboBox,
    QGroupBox, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import QThread, pyqtSignal, QObject

# ================== OCR处理核心逻辑 ==================

class OcrWorker(QObject):
    """
    将OCR处理逻辑放在一个工作线程中，以防止UI冻结。
    """
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, model_path, input_path, output_dir, task, recursive):
        super().__init__()
        self.model_path = model_path
        self.input_path = input_path
        self.output_dir = output_dir
        self.task = task
        self.recursive = recursive
        self.model = None
        self.tokenizer = None
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            self._load_model()
            files_to_process = self._collect_files()
            if not files_to_process:
                self.log_signal.emit("错误：在指定路径下未找到可处理的文件 (.pdf, .png, .jpg, .jpeg, .bmp)。")
                self.finished_signal.emit()
                return

            total_files = len(files_to_process)
            self.log_signal.emit(f"找到 {total_files} 个文件进行处理。")

            for i, file_path in enumerate(files_to_process):
                if not self.is_running:
                    self.log_signal.emit("处理被用户中止。")
                    break
                
                self.log_signal.emit(f"\n{'='*30} [{i+1}/{total_files}] {'='*30}")
                self.log_signal.emit(f"正在处理: {file_path}")

                if str(file_path).lower().endswith('.pdf'):
                    self._process_pdf(file_path)
                else:
                    self._process_image(file_path)
                
                self.progress_signal.emit(int((i + 1) / total_files * 100))

        except Exception as e:
            self.log_signal.emit(f"发生严重错误: {e}")
        finally:
            self.finished_signal.emit()

    def _load_model(self):
        self.log_signal.emit("正在加载模型...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)
            if torch.cuda.is_available():
                self.model = self.model.eval().cuda().to(torch.bfloat16)
                self.log_signal.emit("✓ 模型已加载到 GPU。")
            else:
                self.model = self.model.eval()
                self.log_signal.emit("✓ 模型已加载到 CPU。")
        except Exception as e:
            self.log_signal.emit(f"✗ 模型加载失败: {e}")
            raise

    def _collect_files(self):
        self.log_signal.emit("正在收集文件...")
        source_path = Path(self.input_path)
        valid_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp']
        files = []

        if source_path.is_file():
            if source_path.suffix.lower() in valid_extensions:
                files.append(source_path)
        elif source_path.is_dir():
            if self.recursive:
                for ext in valid_extensions:
                    files.extend(source_path.rglob(f'*{ext}'))
            else:
                 for ext in valid_extensions:
                    files.extend(source_path.glob(f'*{ext}'))
        return files

    def _process_pdf(self, pdf_path):
        start_time = time.time()
        pdf_name = pdf_path.stem
        pdf_output_dir = Path(self.output_dir) / pdf_name
        os.makedirs(pdf_output_dir, exist_ok=True)

        try:
            pdf_doc = fitz.open(pdf_path)
            total_pages = len(pdf_doc)
            self.log_signal.emit(f"PDF共有 {total_pages} 页。")
            all_results = []

            for page_num in range(total_pages):
                if not self.is_running: break
                self.log_signal.emit(f"  - 正在处理第 {page_num + 1}/{total_pages} 页...")
                page = pdf_doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                
                temp_image_path = pdf_output_dir / f'temp_page_{page_num + 1}.png'
                pix.save(temp_image_path)

                page_result_text, _ = self._run_inference(temp_image_path, pdf_output_dir, page_num)
                if page_result_text:
                    all_results.append(f"\n\n# 第 {page_num + 1} 页\n\n{page_result_text}")
                
                os.remove(temp_image_path) # 清理临时图片

            pdf_doc.close()
            
            # 合并所有页结果并保存
            final_text = "\n".join(all_results)
            with open(pdf_output_dir / 'full_result.md', 'w', encoding='utf-8') as f:
                f.write(final_text)
            
            self.log_signal.emit(f"✓ PDF处理完成, 耗时: {time.time() - start_time:.2f}秒")
            self.log_signal.emit(f"  结果保存在: {pdf_output_dir}")

        except Exception as e:
            self.log_signal.emit(f"✗ 处理PDF页面时出错: {e}")

    def _process_image(self, image_path):
        start_time = time.time()
        image_name = image_path.stem
        image_output_dir = Path(self.output_dir) / image_name
        os.makedirs(image_output_dir, exist_ok=True)

        result_text, output_path = self._run_inference(image_path, image_output_dir)
        if result_text:
            self.log_signal.emit(f"✓ 图片处理完成, 耗时: {time.time() - start_time:.2f}秒")
            self.log_signal.emit(f"  结果保存在: {output_path}")

    def _run_inference(self, image_file, output_path, page_num=None):
        prompt = "<image>\n<|grounding|>Convert the document to markdown. " if self.task == 'markdown' else "<image>\nFree OCR. "
        
        try:
            # model.infer的输出路径是目录
            page_output_dir = output_path
            if page_num is not None:
                page_output_dir = output_path / f'page_{page_num + 1}'

            self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=str(image_file),
                output_path=str(page_output_dir),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True
            )

            result_file = page_output_dir / 'result.mmd'
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    page_result = f.read()

                # 如果是PDF页面，处理图片路径
                if page_num is not None:
                    page_images_dir = page_output_dir / 'images'
                    if page_images_dir.exists():
                        # PDF的图片统一存放在根输出目录的images下
                        final_images_dir = Path(self.output_dir) / pdf_path.stem / 'images'
                        os.makedirs(final_images_dir, exist_ok=True)
                        for img_file in os.listdir(page_images_dir):
                            src = page_images_dir / img_file
                            dst = final_images_dir / f'page{page_num + 1}_{img_file}'
                            shutil.copy2(src, dst)
                        # 更新markdown中的图片引用路径
                        page_result = page_result.replace('](images/', f'](images/page{page_num + 1}_')
                        shutil.rmtree(page_images_dir) # 清理单页的images目录

                return page_result, page_output_dir
            else:
                self.log_signal.emit(f"✗ 未找到结果文件: {result_file}")
                return None, None
        except Exception as e:
            self.log_signal.emit(f"✗ 推理失败: {e}")
            return None, None

# ================== PyQt6 UI界面 ==================

class OcrApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DeepSeek-OCR UI')
        self.setGeometry(100, 100, 800, 600)
        self.worker_thread = None
        self.ocr_worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # --- 配置区 ---
        config_group = QGroupBox("配置")
        config_layout = QVBoxLayout()

        # 模型路径
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型路径:"))
        self.model_path_edit = QLineEdit("E:/Ollama/DeepSeek-OCR_models") # 默认路径
        self.model_path_btn = QPushButton("选择...")
        self.model_path_btn.clicked.connect(self.select_model_path)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.model_path_btn)
        config_layout.addLayout(model_layout)

        # 输入路径
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入文件/文件夹:"))
        self.input_path_edit = QLineEdit()
        self.input_file_btn = QPushButton("选择文件...")
        self.input_folder_btn = QPushButton("选择文件夹...")
        self.input_file_btn.clicked.connect(self.select_input_file)
        self.input_folder_btn.clicked.connect(self.select_input_folder)
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(self.input_file_btn)
        input_layout.addWidget(self.input_folder_btn)
        config_layout.addLayout(input_layout)

        # 输出路径
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出文件夹:"))
        self.output_path_edit = QLineEdit("E:/Ollama/output") # 默认路径
        self.output_path_btn = QPushButton("选择...")
        self.output_path_btn.clicked.connect(self.select_output_path)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_path_btn)
        config_layout.addLayout(output_layout)

        # 其他选项
        options_layout = QHBoxLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems(['markdown', 'ocr'])
        options_layout.addWidget(QLabel("任务类型:"))
        options_layout.addWidget(self.task_combo)
        
        self.recursive_radio_group = QButtonGroup(self)
        self.recursive_yes = QRadioButton("递归子文件夹")
        self.recursive_no = QRadioButton("仅当前文件夹")
        self.recursive_yes.setChecked(True)
        self.recursive_radio_group.addButton(self.recursive_yes)
        self.recursive_radio_group.addButton(self.recursive_no)
        options_layout.addWidget(self.recursive_yes)
        options_layout.addWidget(self.recursive_no)
        options_layout.addStretch()
        config_layout.addLayout(options_layout)

        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # --- 控制区 ---
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.stop_btn = QPushButton("停止处理")
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        main_layout.addLayout(control_layout)

        # --- 进度与日志 ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        main_layout.addWidget(QLabel("日志:"))
        main_layout.addWidget(self.log_edit)

        self.setLayout(main_layout)

    def select_model_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型文件夹")
        if path:
            self.model_path_edit.setText(path)

    def select_input_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择输入文件", "", "All Files (*);;PDF Files (*.pdf);;Image Files (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.input_path_edit.setText(path)

    def select_input_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if path:
            self.input_path_edit.setText(path)

    def select_output_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_path_edit.setText(path)

    def append_log(self, message):
        self.log_edit.append(message)

    def start_processing(self):
        model_path = self.model_path_edit.text()
        input_path = self.input_path_edit.text()
        output_dir = self.output_path_edit.text()

        if not all([model_path, input_path, output_dir]):
            self.append_log("错误: 请确保已选择模型、输入和输出路径。")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_edit.clear()

        self.worker_thread = QThread()
        self.ocr_worker = OcrWorker(
            model_path=model_path,
            input_path=input_path,
            output_dir=output_dir,
            task=self.task_combo.currentText(),
            recursive=self.recursive_yes.isChecked()
        )
        self.ocr_worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.ocr_worker.run)
        self.ocr_worker.finished_signal.connect(self.on_processing_finished)
        self.ocr_worker.log_signal.connect(self.append_log)
        self.ocr_worker.progress_signal.connect(self.progress_bar.setValue)

        self.worker_thread.start()

    def stop_processing(self):
        if self.ocr_worker:
            self.ocr_worker.stop()
            self.append_log("正在发送停止信号...")
            self.stop_btn.setEnabled(False)

    def on_processing_finished(self):
        self.append_log("\n处理全部完成。")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.worker_thread = None
        self.ocr_worker = None

    def closeEvent(self, event):
        self.stop_processing()
        super().closeEvent(event)

if __name__ == '__main__':
    # 设置CUDA设备环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    app = QApplication(sys.argv)
    ex = OcrApp()
    ex.show()
    sys.exit(app.exec())