import sys
import time

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

from strategy import *
from ui import CalculatingWindow, DashboardWindow
from datetime import datetime
import os


# 계산 작업을 수행하는 QThread 기반 Worker
class CalculationWorker(QThread):
    calculation_done = pyqtSignal()
    progress_update = pyqtSignal(str)
    progress_percent = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.data = {}
        self.text = ""

    def run(self):
        try:
            self.progress_update.emit("Fetching data...")
            self.progress_percent.emit(0)

            chart, month_chart = fetch_charts()

            self.progress_update.emit("Backtesting...")
            self.progress_percent.emit(30)

            strategy = InvestmentStrategyIntegration(chart=chart, month_chart=month_chart)
            portfolio = strategy.portfolio
            portfolio = portfolio.apply(lambda x: pd.Series(x).value_counts(normalize=True).to_dict())

            ref_date = portfolio.index[-1].date()
            weights_str = ", ".join([f"{key}={value * 100:.2f}%" for key, value in portfolio.iloc[-1].items()])
            output = f"{ref_date} final portfolio: {weights_str}"
            logger.info(output)

            self.data = {k: v * 100 for k, v in portfolio.iloc[-1].items()}

            self.progress_update.emit("Making backtest reports...")
            self.progress_percent.emit(80)

            strategy.backtest("./output/final")
            logger.info("Backtest completed successfully.")

            self.text = f"""
    Created Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Reference Date: {ref_date}
            """

            self.progress_update.emit("Making backtest reports...")
            self.progress_percent.emit(100)
            self.progress_update.emit("Done.")
            time.sleep(1)
            self.calculation_done.emit()
        except BaseException as e:
            logger.error(msg="An error occurs while running worker.", exc_info=e)


def main():
    os.makedirs("output", exist_ok=True)

    app = QApplication(sys.argv)
    app.setApplicationName("Asset Allocation")

    # 진행 창 표시
    progress_window = CalculatingWindow()
    progress_window.setGeometry(100, 100, 1600, 900)
    progress_window.show()

    # 계산 Worker 시작
    worker = CalculationWorker()
    worker.progress_update.connect(progress_window.message_label.setText)
    worker.progress_percent.connect(progress_window.progress_bar.setValue)
    worker.calculation_done.connect(lambda: on_calculation_finished(progress_window, worker.data, worker.text))
    worker.start()

    logger.info("Calling app.exec()")
    result = app.exec()
    logger.info("Calling sys.exit(result)")
    sys.exit(result)


def on_calculation_finished(progress_window, data, text):
    geometry = progress_window.geometry()  # 현재 창 크기 및 위치 정보를 가져옴
    progress_window.close()
    result_window = DashboardWindow(data, text)
    result_window.setGeometry(geometry)  # 동일한 geometry 적용
    result_window.show()

    global main_result_window
    main_result_window = result_window


if __name__ == "__main__":
    try:
        logger.info("Calling main()")
        main()
        logger.info("main() finished.")
    except BaseException as _e:
        logger.error(msg="An error occurs while running main()", exc_info=_e)
