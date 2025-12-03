from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
import sys
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image as PILImage
from scripts.model import Model

def video_capture_to_pixmap(capture):
    ret, frame = capture.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        return pixmap
    else:
        return None

class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout_vbox = QVBoxLayout(self)

        self.label_video = QLabel()
        self.label_video.setFixedSize(512, 512)
        self.label_video.setStyleSheet("background-color: black;")
        self.label_status = QLabel("waiting for video...")

        self.layout_vbox.addWidget(self.label_video)
        self.layout_vbox.addWidget(self.label_status)

        self.executor = ThreadPoolExecutor(max_workers=1)

        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)

        if not self.capture.isOpened():
            self.label_status.setText("Error: Could not open video device.")
        
        self.timer_preview = QTimer()
        self.timer_preview.timeout.connect(self.update_preview)
        self.timer_preview.start(66) # 15 fps

    def update_preview(self):
        pixmap = video_capture_to_pixmap(self.capture)
        if pixmap:
            self.label_video.setPixmap(pixmap.scaled(self.label_video.size(), Qt.KeepAspectRatioByExpanding))
            self.label_status.setText("Video streaming...")
        else:
            self.label_status.setText("Error: Could not read frame from video device.")

    def grab_PILImage(self) -> PILImage:
        ret, frame = self.capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return PILImage.fromarray(frame_rgb)
        else:
            return None



class classifierGUI(QWidget):
    def __init__(self, model: Model, capture_interval=1000, parent=None):
        super().__init__(parent)

        self.capture_interval = capture_interval
        self.model = model

        self.layout_vbox = QVBoxLayout(self)

        self.setWindowTitle("Waste Classifier")
        self.setGeometry(100, 100, 800, 600)

        self.executor = ThreadPoolExecutor(max_workers=1)


        # widgets
        self.video_widget = VideoWidget(self)
        self.layout_vbox.addWidget(self.video_widget)

        self.label_result = QLabel("Classification Result: N/A")
        self.layout_vbox.addWidget(self.label_result)

        self.setLayout(self.layout_vbox)

        # timer for capturing frames
        self.timer_capture = QTimer()
        self.timer_capture.timeout.connect(self.capture_and_classify)
        self.timer_capture.start(self.capture_interval)

    def capture_and_classify(self):
        image = self.video_widget.grab_PILImage()
        if image is not None:
            # Assuming you have a model instance named self.model
            result = self.model.classify(image)
            self.label_result.setText(f"Classification Result: {result}")
        else:
            self.label_result.setText("Error: Could not capture frame.")

def run_GUI(model: Model):
    app = QApplication(sys.argv)
    gui = classifierGUI(model)
    gui.show()
    sys.exit(app.exec())