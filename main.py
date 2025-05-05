import sys
from PyQt5.QtWidgets import QApplication
from ui import WeatherApp, LoginWindow
from database import init_db, init_users

class AppController:
    def __init__(self):
        self.app = QApplication(sys.argv)
        init_db()
        init_users()
        self.login_window = LoginWindow(on_success_callback=self.launch_main)
        self.login_window.show()

    def launch_main(self):
        self.main_window = WeatherApp()
        self.main_window.show()

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    controller = AppController()
    controller.run()