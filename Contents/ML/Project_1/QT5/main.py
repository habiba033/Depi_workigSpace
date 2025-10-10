import sys
from PyQt5 import QtWidgets
from app.main_window import HousingApp
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = HousingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()