from PyQt5.QtWidgets import QApplication
import sys
from ui import WeatherApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WeatherApp()
    window.show()
    sys.exit(app.exec_())
