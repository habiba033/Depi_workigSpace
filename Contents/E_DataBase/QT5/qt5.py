from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QTableWidgetItem
)
from PyQt5 import uic
import sys
import psycopg2


class Main_App(QMainWindow):
    def __init__(self):
        super(Main_App, self).__init__()
        uic.loadUi(r'Contents\E_DataBase\QT5\std.ui', self)
        self.tabWidget.tabBar().setVisible(False)
        self.InitUI()
        self.handle_db_conn()
        self.handle_btn()
        self.load_students()  

    def InitUI(self):
        self.setWindowTitle("Student System")

    def handle_db_conn(self):
        try:
            self.db = psycopg2.connect(
                dbname="GoBike",
                user="postgres",
                password="habiba2004#",
                host="localhost",
                port="5433"
            )
            self.curr = self.db.cursor()
            print("Connection is Done!")
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Failed to connect:\n{str(e)}")

    def handle_btn(self):
        self.std_add_btn.clicked.connect(self.add_std_info)
        self.std_update_btn.clicked.connect(self.update_std_info)
        self.std_del_btn.clicked.connect(self.del_std_info)
        # self.std_view_btn.clicked.connect(self.load_students)

    def add_std_info(self):
        try:
            std_id = self.std_Id_txt.text()
            std_name = self.std_name_txt.text()
            std_email = self.std_email_txt.text()
            std_phone = self.std_phone_txt.text()

            self.curr.execute(
                'INSERT INTO students (student_id, name, email, phone) VALUES(%s,%s,%s,%s)',
                (std_id, std_name, std_email, std_phone)
            )
            self.db.commit()
            QMessageBox.information(self, "Success", "Student added successfully!")
            self.load_students()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add student:\n{str(e)}")

    def update_std_info(self):
        try:
            std_id = self.std_Id_txt.text()
            new_name = self.std_name_txt.text()
            new_email = self.std_email_txt.text()
            new_phone = self.std_phone_txt.text()

            self.curr.execute(
                "UPDATE students SET name=%s, email=%s, phone=%s WHERE student_id=%s",
                (new_name, new_email, new_phone, std_id)
            )
            self.db.commit()
            QMessageBox.information(self, "Success", "Student updated successfully!")
            self.load_students()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Update failed:\n{str(e)}")

    def del_std_info(self):
        try:
            std_id = self.std_Id_txt.text()
            self.curr.execute("DELETE FROM students WHERE student_id=%s", (std_id,))
            self.db.commit()
            QMessageBox.information(self, "Success", "Student deleted successfully!")
            self.load_students()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Delete failed:\n{str(e)}")

    def load_students(self):
        """عرض كل الطلاب في TableWidget"""
        try:
            self.curr.execute("SELECT student_id, name, email, phone FROM students")
            rows = self.curr.fetchall()

            self.std_table.setRowCount(0)  # مسح القديم
            for row_number, row_data in enumerate(rows):
                self.std_table.insertRow(row_number)
                for column_number, data in enumerate(row_data):
                    self.std_table.setItem(
                        row_number, column_number, QTableWidgetItem(str(data))
                    )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load students:\n{str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Main_App()
    window.show()
    sys.exit(app.exec_())