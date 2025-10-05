
from PyQt5.QtWidgets import QMessageBox


# ---------- Custom Message ----------
def show_message(parent, title, text, success=True):
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)

    if success:
        msg.setIcon(QMessageBox.Information)
        msg.setStyleSheet("QLabel{color: green; font-size: 14px;} QPushButton{min-width:80px;}")
    else:
        msg.setIcon(QMessageBox.Critical)
        msg.setStyleSheet("QLabel{color: red; font-size: 14px;} QPushButton{min-width:80px;}")

    msg.exec_()
