import sys
import os
from PyQt5 import uic
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget,QTableWidgetItem
from PyQt5.QtWidgets import QHeaderView
# أضف root folder إلى sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from db import get_connection
from utils import QMessageBox

#  ---------- View Users Window ----------

class ViewUsersWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ui", "view_users.ui")
        uic.loadUi(ui_path, self)

        
        self.usersTable.setColumnCount(5)
        self.usersTable.setHorizontalHeaderLabels(["ID", "Birth Year", "Age", "Gender", "User Type"])
        header = self.usersTable.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.searchInput.setPlaceholderText("Search By ID: ")

        # ربط البحث بالتغيير في النص
        self.searchInput.textChanged.connect(self.search_user)

        # تحميل كل المستخدمين عند الفتح
        self.load_all_users()

    def load_all_users(self):
        """عرض كل المستخدمين في الجدول"""
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT user_id, birth_year, age, gender, user_type FROM dim_user ORDER BY user_id")
            rows = cur.fetchall()
            cur.close()
            conn.close()

            self.show_in_table(rows)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load users: {e}")

    def show_in_table(self, rows):
        """عرض بيانات في الجدول"""
        self.usersTable.setRowCount(0)
        for row in rows:
            row_idx = self.usersTable.rowCount()
            self.usersTable.insertRow(row_idx)
            for col_idx, value in enumerate(row):
                self.usersTable.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

    def search_user(self, text):
        """بحث مباشر بمجرد الكتابة"""
        search_term = text.strip()

        try:
            conn = get_connection()
            cur = conn.cursor()

            if search_term == "":
                # لو السيرش فاضي رجّع كل البيانات
                cur.execute("SELECT user_id, birth_year, age, gender, user_type FROM dim_user ORDER BY user_id")
            elif search_term.isdigit():
                # بحث بالـ ID
                cur.execute("SELECT user_id, birth_year, age, gender, user_type FROM dim_user WHERE user_id = %s", (search_term,))
            else:
                # ممكن تبحثي بالـ user_type أو gender (حسب ما تحبي)
                cur.execute("SELECT user_id, birth_year, age, gender, user_type FROM dim_user WHERE user_type ILIKE %s OR gender ILIKE %s", 
                            (f"%{search_term}%", f"%{search_term}%"))

            rows = cur.fetchall()
            cur.close()
            conn.close()

            self.show_in_table(rows)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Search failed: {e}")
