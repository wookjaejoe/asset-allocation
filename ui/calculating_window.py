from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QProgressBar
from PyQt6.QtCore import Qt


class CalculatingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calculation")

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.wait_label = QLabel("Please wait a moment...", self)
        self.wait_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.wait_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumWidth(500)
        self.progress_bar.setMaximumWidth(500)
        layout.addWidget(self.progress_bar)

        self.message_label = QLabel("계산 시작 중...", self)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message_label)
