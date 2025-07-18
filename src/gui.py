# gui.py (v3.5 - 添加了重置功能)
import sys
import warnings
import time
import cv2
import numpy as np
import tracking_logic as tracker
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, 
                             QVBoxLayout, QMessageBox, QMainWindow, QStatusBar,
                             QListWidget, QListWidgetItem, QFrame, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer

warnings.filterwarnings('ignore')

# --- PyQt5 Worker and Widgets ---

class VideoWorker(QThread):
    new_frame = pyqtSignal(int, np.ndarray)
    new_tracker_view = pyqtSignal(int, object)
    active_ids_update = pyqtSignal(int, set)
    status_update = pyqtSignal(str)
    
    def __init__(self, cam_id, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self._is_running = True

    def run(self):
        try:
            cam_generator = tracker.process_camera(self.cam_id)
            while self._is_running:
                result = next(cam_generator)
                if result.get("main_frame") is not None:
                    self.new_frame.emit(self.cam_id, result["main_frame"])
                if result.get("active_sids") is not None:
                    self.active_ids_update.emit(self.cam_id, result["active_sids"])
                
                view_to_show = result.get("tracker_view")
                should_close = result.get("should_close_tracker")
                if should_close:
                    self.new_tracker_view.emit(self.cam_id, None)
                else:
                    self.new_tracker_view.emit(self.cam_id, view_to_show)
        except StopIteration:
            self.status_update.emit(f"摄像头 {self.cam_id} 视频流结束。")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.status_update.emit(f"错误: CAM {self.cam_id} 线程崩溃: {e}")

    def stop(self):
        self._is_running = False
        self.wait(2000)

class AspectRatioLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._pixmap = QPixmap()

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        if self._pixmap.isNull():
            super().paintEvent(event)
            return
        
        scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        from PyQt5.QtGui import QPainter
        painter = QPainter(self)
        x = (self.width() - scaled_pixmap.width()) / 2
        y = (self.height() - scaled_pixmap.height()) / 2
        painter.drawPixmap(int(x), int(y), scaled_pixmap)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("双摄像头协同追踪系统")
        self.setGeometry(50, 50, 1600, 900)
        self.setStyleSheet("background-color: white; color: black;")

        self.threads = {}
        self.is_running = False
        self.active_sids = {}

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, 75)

        preview_panel = QWidget()
        preview_layout = QHBoxLayout(preview_panel)
        
        self.cam0_label = AspectRatioLabel("CAM 0")
        self.cam2_label = AspectRatioLabel("CAM 2")
        for label in [self.cam0_label, self.cam2_label]:
            label.setMinimumSize(320, 180)
            label.setStyleSheet("border: 1px solid #AAAAAA; background-color: #F0F0F0;")
            label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.cam0_label)
        preview_layout.addWidget(self.cam2_label)
        
        self.tracker_label = AspectRatioLabel("Tracker View")
        self.tracker_label.setMinimumSize(640, 360)
        self.tracker_label.setStyleSheet("border: 2px solid #007ACC; background-color: #F0F0F0;")
        self.tracker_label.setAlignment(Qt.AlignCenter)

        left_layout.addWidget(preview_panel)
        left_layout.addWidget(self.tracker_label)

        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, 25)

        id_list_label = QLabel("当前识别ID列表")
        id_list_label.setFont(QFont("Arial", 12))
        self.id_list_widget = QListWidget()
        self.id_list_widget.setStyleSheet(
            "QListWidget { border: 1px solid #AAAAAA; font-size: 16px; color: black; }"
            "QListWidget::item:selected { background-color: #007ACC; color: white; }"
        )

        button_panel = QWidget()
        button_layout = QHBoxLayout(button_panel)
        self.start_btn = QPushButton("🚀启动追踪")
        self.stop_btn = QPushButton("🛑停止追踪")
        # <<< 新增: 创建重置按钮 >>>
        self.reset_btn = QPushButton("🔄重置系统")
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        for btn in [self.start_btn, self.stop_btn, self.reset_btn]:
            btn.setMinimumHeight(40); btn.setStyleSheet("padding: 5px;")
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.reset_btn)

        right_layout.addWidget(id_list_label); right_layout.addWidget(self.id_list_widget)
        right_layout.addStretch(); right_layout.addWidget(button_panel)

        self.status_bar = QStatusBar(self); self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("准备就绪。")

        self.id_cleanup_timer = QTimer(self)
        self.id_cleanup_timer.timeout.connect(self.refresh_id_list)
        
        self.start_btn.clicked.connect(self.start_tracking)
        self.stop_btn.clicked.connect(self.stop_tracking)
        # <<< 新增: 连接重置按钮的点击信号 >>>
        self.reset_btn.clicked.connect(self.reset_tracking)
        self.id_list_widget.itemClicked.connect(self.on_id_list_clicked)

    def start_tracking(self):
        if self.is_running: return
        self.active_sids.clear()
        for cam_id in [0, 2]:
            worker = VideoWorker(cam_id, self)
            worker.new_frame.connect(self.update_video_frame)
            worker.new_tracker_view.connect(self.update_tracker_view)
            worker.active_ids_update.connect(self.update_active_sids)
            worker.status_update.connect(self.update_status)
            worker.start()
            self.threads[cam_id] = worker
        self.is_running = True
        self.id_cleanup_timer.start(1000)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        # <<< 新增: 启动时激活重置按钮 >>>
        self.reset_btn.setEnabled(True)
        self.statusBar().showMessage("追踪已启动...")

    def stop_tracking(self, is_resetting=False):
        if not self.is_running: return
        self.id_cleanup_timer.stop()
        for thread in self.threads.values(): thread.stop()
        self.threads.clear()
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # <<< 新增: 停止时禁用重置按钮 >>>
        self.reset_btn.setEnabled(False)
        
        self.cam0_label.clear(); self.cam0_label.setText("CAM 0"); self.cam0_label.setAlignment(Qt.AlignCenter)
        self.cam2_label.clear(); self.cam2_label.setText("CAM 2"); self.cam2_label.setAlignment(Qt.AlignCenter)
        self.tracker_label.clear(); self.tracker_label.setText("Tracker View"); self.tracker_label.setAlignment(Qt.AlignCenter)
        self.id_list_widget.clear(); self.active_sids.clear()
        
        if not is_resetting:
            self.statusBar().showMessage("追踪已停止。系统已恢复初始状态。")

    # <<< 新增: 重置按钮的槽函数 >>>
    def reset_tracking(self):
        """停止追踪，重置后端状态，并刷新UI"""
        self.statusBar().showMessage("正在重置系统...")
        # 1. 停止当前追踪并重置UI (传入True避免重复显示状态信息)
        self.stop_tracking(is_resetting=True)
        
        # 2. 调用后端的重置函数
        tracker.reset_state()
        
        # 3. 更新状态栏
        self.statusBar().showMessage("系统已重置，准备就绪。")

    def update_video_frame(self, cam_id, cv_img):
        label = self.cam0_label if cam_id == 0 else self.cam2_label
        qt_img = self._convert_cv_qt(cv_img)
        label.setPixmap(qt_img)

    def update_tracker_view(self, cam_id, frame_or_none):
        controlling_cam = tracker.current_cam
        if frame_or_none is None:
            if cam_id == controlling_cam or controlling_cam == -1:
                self.tracker_label.clear(); self.tracker_label.setText("Tracker View")
            return
        if cam_id == controlling_cam:
            qt_img = self._convert_cv_qt(frame_or_none)
            self.tracker_label.setPixmap(qt_img)
        
    def update_active_sids(self, cam_id, sids):
        now = time.time()
        for sid in sids: self.active_sids[sid] = now

    def refresh_id_list(self):
        now = time.time()
        self.active_sids = {sid: t for sid, t in self.active_sids.items() if now - t < 3.0}
        current_target = tracker.target_id
        self.id_list_widget.clear()
        for sid in sorted(self.active_sids.keys()):
            item = QListWidgetItem(f"ID: {sid}")
            if sid == current_target:
                item.setFont(QFont("Arial", 12, QFont.Bold)); item.setBackground(QColor("#007ACC")); item.setForeground(QColor("white"))
            self.id_list_widget.addItem(item)
            
    def on_id_list_clicked(self, item):
        try:
            clicked_sid = int(item.text().split(': ')[1])
            current_target = tracker.target_id
            if clicked_sid == current_target:
                tracker.clear_target()
                self.statusBar().showMessage(f"已取消追踪 ID: {clicked_sid}")
            else:
                tracker.set_target(clicked_sid)
                self.statusBar().showMessage(f"已设置追踪目标为 ID: {clicked_sid}")
            self.refresh_id_list()
        except (ValueError, IndexError): pass
    
    def update_status(self, message):
        self.status_bar.showMessage(message)

    def _convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_format)

    def closeEvent(self, event):
        self.statusBar().showMessage("正在关闭...")
        self.stop_tracking()
        tracker.cleanup_and_save()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        tracker.initialize()
    except Exception as e:
        QMessageBox.critical(None, "初始化失败", f"加载模型或初始化追踪器时发生致命错误:\n\n{e}\n\n请检查您的模型文件和环境。")
        sys.exit(1)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())