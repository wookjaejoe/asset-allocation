from ftplib import print_line

from PyQt6.QtCharts import QChart, QChartView, QPieSeries
from PyQt6.QtCore import Qt, QMargins
from PyQt6.QtGui import QPainter, QFont, QBrush, QColor, QIntValidator
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QLineEdit

colors = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#edc949",
    "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab", "#6b6ecf",
    "#b5bd61", "#17becf", "#bcbd22", "#8c564b", "#ff9896",
]


class DashboardWindow(QMainWindow):
    def __init__(
            self,
            data: dict,
            text: str
    ):
        super().__init__()
        self.setWindowTitle("Dashboard")
        self.setStyleSheet("background-color: #2E2E2E;")  # 창 배경색 (어두운 테마)

        self.data = data
        self.text = text
        self.text_input = None
        self.new_text = None

        # 메인 레이아웃 (수평 배치)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 왼쪽: 차트
        self.chart_view = self.create_chart()
        main_layout.addWidget(self.chart_view, 2)  # 차트가 2:1 비율로 큼

        # 오른쪽: ETF 비율 텍스트 & 사용자 입력
        self.text_widget = self.create_scrollable_text()
        main_layout.addWidget(self.text_widget, 1)  # ETF 비율 텍스트

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_chart(self):
        """도넛 차트 생성"""
        series = QPieSeries()
        series.setHoleSize(0.5)  # 도넛 스타일
        label_font = QFont("Arial", 13, QFont.Weight.Medium)  # ✅ 글자 크기 고정

        for i, (name, value) in enumerate(self.data.items()):
            slice_ = series.append(name, value)
            slice_.setBrush(QBrush(QColor(colors[i % len(colors)])))
            slice_.setLabel(f"{name} ({value:.2f}%)")
            slice_.setLabelVisible(True)
            slice_.setLabelFont(label_font)

            # ✅ 라벨 색상을 배경 밝기에 따라 조정
            background_color = self.palette().color(self.backgroundRole())
            brightness = self.calculate_brightness(QColor(background_color))

            if brightness < 128:  # 어두운 배경일 경우 라벨을 밝은 색으로
                slice_.setLabelColor(QColor("lightgrey"))
            else:  # 밝은 배경일 경우 라벨을 어두운 색으로
                slice_.setLabelColor(QColor("darkgrey"))

        # ✅ QChart 설정 (테두리 제거)
        chart = QChart()
        chart.addSeries(series)
        chart.legend().setVisible(False)
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        chart.setBackgroundBrush(QBrush(Qt.GlobalColor.transparent))
        chart.setMargins(QMargins(0, 0, 0, 0))

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        chart_view.setStyleSheet("border-radius: 10px; background: transparent;")  # ✅ 둥근 모서리 적용

        return chart_view

    def create_scrollable_text(self):
        """좌우 스크롤이 가능한 텍스트 영역 + 숫자 입력 필드 추가"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #1E1E1E; border-radius: 10px;")

        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # ✅ 최상단 정렬 추가

        # 제목
        title = QLabel("ETF Allocation Details")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: white; margin-bottom: 10px;")
        layout.addWidget(title)

        # 텍스트
        text_label = QLabel(self.text)
        text_label.setFont(QFont("Arial", 12))
        text_label.setWordWrap(True)
        text_label.setStyleSheet("color: lightgrey;")
        layout.addWidget(text_label)

        # ★ 사용자로부터 숫자를 입력받는 필드 추가
        input_label = QLabel("Total investment amount:")
        input_label.setFont(QFont("Arial", 12))
        input_label.setStyleSheet("color: lightgrey; margin-top: 10px;")
        layout.addWidget(input_label)

        # QLineEdit 생성 및 숫자 전용 Validator 설정
        self.number_input = QLineEdit()
        self.number_input.setFont(QFont("Arial", 12))
        self.number_input.setStyleSheet("background-color: #FFFFFF; color: black; padding: 5px;")
        self.number_input.setPlaceholderText("Only integers allowed")
        self.number_input.textChanged.connect(self.update_label)

        # IntValidator 또는 DoubleValidator로 숫자만 입력 받도록 설정 가능
        # 예) 정수만 입력받도록 제한
        validator = QIntValidator()
        self.number_input.setValidator(validator)
        layout.addWidget(self.number_input)

        self.new_text = QLabel("Please enter the total investment amount in the correct format.")
        self.new_text.setFont(QFont("Arial", 12))
        self.new_text.setWordWrap(True)
        self.new_text.setStyleSheet("color: lightgrey;")
        layout.addWidget(self.new_text)

        container.setLayout(layout)
        scroll_area.setWidget(container)
        return scroll_area

    def update_label(self):
        user_text = self.number_input.text()

        try:
            amount = int(user_text)
            text = f"""
{"\n".join([f"{k} = {round(amount * v / 100):,}" for k, v in self.data.items()])}
"""
        except:
            text = "Please enter the total investment amount in the correct format."

        self.new_text.setText(text)

    @staticmethod
    def calculate_brightness(color):
        return color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
