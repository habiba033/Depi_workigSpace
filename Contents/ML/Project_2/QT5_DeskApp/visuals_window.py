import numpy as np
# --- ADDED: Imports from visuals_window.py ---
from PyQt5.QtWidgets import ( QMainWindow, QWidget, 
                             QVBoxLayout, QHBoxLayout, QLabel, 
                             QFrame, QScrollArea, QSpacerItem, 
                             QSizePolicy, QGraphicsDropShadowEffect,
                             QGridLayout, QTabWidget, QTableWidget,
                             QTableWidgetItem)
from PyQt5.QtChart import (QChart, QChartView, QBarSeries, QBarSet, 
                           QValueAxis, QBarCategoryAxis, QLineSeries,
                           QSplineSeries, QHorizontalBarSeries)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QFont, QColor
from predictor import PredictWindow

QSS_STYLES = """
    /* Overall window background (like HTML body) */
    QMainWindow {
        background-color: #f0f4f8; 
    }

    /* Main Page Title */
    QLabel#pageTitle {
        font-size: 28px; /* text-3xl */
        font-weight: 800; /* font-extrabold */
        color: #1f2937; /* text-gray-800 */
        padding-bottom: 8px;
        border-bottom: 1px solid #e5e7eb; /* border-b */
        margin-bottom: 20px; /* mb-8 */
    }
    
    /* --- Tab Widget Styling --- */
    QTabWidget::pane {
        border: 1px solid #e5e7eb; /* border-gray-200 */
        border-top: none;
        background-color: #ffffff;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
    }
    QTabWidget::tab-bar { left: 5px; }
    
    QTabBar::tab {
        width: 200px;
        background: #f9fafb; /* bg-gray-50 */
        border: 1px solid #e5e7eb;
        border-bottom: none; 
        
        padding: 14px 48px;  /* Big padding */
        font-size: 14px;     /* Big font */
        
        font-weight: 500;
        color: #4b5563; /* text-gray-600 */
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 5px; 
    }
    
    QTabBar::tab:hover {
        background: #f3f4f6; /* bg-gray-100 */
    }
    QTabBar::tab:selected {
        background: #ffffff; /* white */
        color: #0ea5e9; /* text-sky-500 */
        font-weight: bold;
        border: 1px solid #e5e7eb;
        border-bottom: 2px solid #ffffff; 
        margin-bottom: -1px; 
    }
    
    /* Scroll Area Styling */
    QScrollArea { border: none; background-color: transparent; }
    QWidget#scrollContent { background-color: transparent; }
    QScrollBar:vertical {
        border: none; background: #e2e8f0;
        width: 10px; margin: 0; border-radius: 5px;
    }
    QScrollBar::handle:vertical {
        background: #94a3b8; min-height: 20px; border-radius: 5px;
    }

    /* --- ADDING LIFE! --- */
    
    /* 1. Card "Glow" on Hover */
    QFrame#card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb; /* default border */
        border-radius: 12px;
        padding: 24px;
        /* Add a transition for a smooth effect */
        transition: border 0.2s ease-in-out;
    }
    QFrame#card:hover {
        border: 1px solid #0ea5e9; /* "Glow" with the theme color */
    }
    
    /* Card Titles */
    QLabel#cardTitle {
        font-size: 20px; font-weight: 600;
        color: #374151; padding-bottom: 4px;
    }
    QLabel#cardSubtitle {
        font-size: 14px; color: #4b5563; padding-bottom: 16px;
    }
"""


class VizWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualization & Model Insights")
        self.setGeometry(100, 100, 1200, 900)
        self.setStyleSheet(QSS_STYLES)
        
        # --- Define reusable fonts for charts ---
        self.chart_title_font = QFont("Arial", 14, QFont.Bold)
        self.chart_axis_title_font = QFont("Arial", 11, QFont.Bold)
        self.chart_label_font = QFont("Arial", 10)
        
        # --- Create the main page widget ---
        main_page_widget = QWidget()
        page_layout = QVBoxLayout(main_page_widget)
        page_layout.setContentsMargins(24, 24, 24, 24) 
        
        title = QLabel("📊 Visualization & Model Insights", objectName="pageTitle")
        page_layout.addWidget(title)

        self.tab_widget = QTabWidget()
        page_layout.addWidget(self.tab_widget) 

        tab_demo = self.create_demographics_tab()
        tab_corr = self.create_correlations_tab()
        tab_model = self.create_model_diagnostics_tab()

        self.tab_widget.addTab(tab_demo, "Demographics & Trends")
        self.tab_widget.addTab(tab_corr, "Correlations & Distributions")
        self.tab_widget.addTab(tab_model, "Model Diagnostics")
        
        self.setCentralWidget(main_page_widget)

    # -----------------------------------------------------------------
    #  TAB PAGE CREATION (Unchanged logic)
    # -----------------------------------------------------------------

    def create_demographics_tab(self):
        scroll_area, scroll_layout = self.create_scrollable_tab_page()
        
        layout1 = QVBoxLayout()
        layout1.addWidget(QLabel("📉 Income Trends by Age", objectName="cardTitle"))
        layout1.addWidget(self.create_income_trends_chart())
        scroll_layout.addWidget(self.create_card(layout1))

        layout2 = QVBoxLayout()
        layout2.addWidget(QLabel("👥 Demographic Insights", objectName="cardTitle"))
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.create_income_by_education_chart())
        h_layout.addWidget(self.create_income_by_gender_chart())
        layout2.addLayout(h_layout)
        scroll_layout.addWidget(self.create_card(layout2))
        
        scroll_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return scroll_area

    def create_correlations_tab(self):
        scroll_area, scroll_layout = self.create_scrollable_tab_page()

        layout1 = QVBoxLayout()
        layout1.addWidget(QLabel("🌡️ Correlation Heatmap (Numerical Features)", objectName="cardTitle"))
        layout1.addWidget(self.create_correlation_heatmap())
        scroll_layout.addWidget(self.create_card(layout1))

        layout2 = QVBoxLayout()
        layout2.addWidget(QLabel("📈 Skewed Feature Distribution (Before/After Log Transform)", objectName="cardTitle"))
        layout2.addWidget(QLabel("Compare raw vs. processed data to justify preprocessing steps.", objectName="cardSubtitle"))
        layout2.addWidget(self.create_distribution_charts())
        scroll_layout.addWidget(self.create_card(layout2))
        

        scroll_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return scroll_area

    def create_model_diagnostics_tab(self):
        scroll_area, scroll_layout = self.create_scrollable_tab_page()
        
        layout1 = QVBoxLayout()
        layout1.addWidget(QLabel("🔑 Top 10 Feature Importances", objectName="cardTitle"))
        layout1.addWidget(QLabel("The core drivers for predicting income.", objectName="cardSubtitle"))
        layout1.addWidget(self.create_feature_importance_horizontal_chart())
        scroll_layout.addWidget(self.create_card(layout1))

        layout2 = QVBoxLayout()
        layout2.addWidget(QLabel("🔬 Model Assessment", objectName="cardTitle"))
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.create_roc_curve_chart())
        h_layout.addWidget(self.create_confusion_matrix())
        layout2.addLayout(h_layout)
        scroll_layout.addWidget(self.create_card(layout2))

        scroll_layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return scroll_area

    # -----------------------------------------------------------------
    #  HELPER WIDGETS (Unchanged)
    # -----------------------------------------------------------------
    
    def create_scrollable_tab_page(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }") 
        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(24)
        scroll_layout.setContentsMargins(20, 20, 25, 20) 
        scroll_area.setWidget(scroll_content)
        return scroll_area, scroll_layout
    
    def create_card(self, content_layout):
        card = QFrame()
        card.setObjectName("card")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 5)
        card.setGraphicsEffect(shadow)
        card.setLayout(content_layout)
        return card

    def create_placeholder(self, text):
        label = QLabel(text)
        label.setFont(QFont("Arial", 14))
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(300)
        label.setFrameShape(QFrame.StyledPanel)
        label.setStyleSheet("border: 2px dashed #aaa; border-radius: 8px;")
        return label

    # -----------------------------------------------------------------
    #  CHART CREATION FUNCTIONS (Unchanged)
    # -----------------------------------------------------------------
    
    def create_correlation_heatmap(self):
        data = [
            [1.00, 0.04, 0.08, 0.06, 0.10, 0.24],
            [0.04, 1.00, 0.13, -0.03, 0.08, 0.15],
            [0.08, 0.13, 1.00, -0.03, -0.05, 0.08],
            [0.06, -0.03, -0.03, 1.00, 0.05, 0.15],
            [0.10, 0.08, -0.05, 0.05, 1.00, 0.22],
            [0.24, 0.15, 0.08, 0.15, 0.22, 1.00]
        ]
        labels = ["age", "edu-num", "capital-gain", "capital-loss", "hours/wk", "donation"]

        table = QTableWidget(len(labels), len(labels))
        table.setHorizontalHeaderLabels(labels)
        table.setVerticalHeaderLabels(labels)

        header_font = QFont("Arial", 10, QFont.Bold)
        table.horizontalHeader().setFont(header_font)
        table.verticalHeader().setFont(header_font)
        table.setFont(self.chart_label_font)

        for i in range(len(labels)):
            for j in range(len(labels)):
                value = data[i][j]
                item = QTableWidgetItem(f"{value:.2f}")
                item.setTextAlignment(Qt.AlignCenter)

                if value == 1.0: color = QColor("#67001f") # Dark Red
                elif value > 0.5: color = QColor("#d6604d")
                elif value > 0.1: color = QColor("#f4a582")
                elif value >= 0: color = QColor("#fddbc7")
                elif value > -0.1: color = QColor("#d1e5f0")
                elif value > -0.5: color = QColor("#92c5de")
                else: color = QColor("#053061") # Dark Blue

                item.setBackground(color)
                if value == 1.0 or value <= -0.5:
                     item.setForeground(QColor("white"))

                table.setItem(i, j, item)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        # --- MODIFIED: Increased Minimum Height ---
        table.setMinimumHeight(500) # Was 350
        # --- END MODIFICATION ---

        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return table

    def create_feature_importance_horizontal_chart(self):
        series = QHorizontalBarSeries()
        data = [("native-country_United-States", 0.01), ("race_White", 0.02),
                ("occupation_Exec-managerial", 0.03), ("capital-loss", 0.05),
                ("relationship_Husband", 0.08), ("hours-per-week", 0.12),
                ("marital-status_Married-civ-spouse", 0.13), ("capital-gain", 0.14),
                ("education-num", 0.15), ("age", 0.25)]
        
        bar_set = QBarSet("Feature Importance")
        categories = []
        for item in data:
            categories.append(item[0])
            bar_set.append(item[1])
        
        bar_set.setColor(QColor("#c026d3")) # Fuchsia/Purple
        series.append(bar_set)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitleFont(self.chart_title_font)
        
        chart.setAnimationOptions(QChart.AllAnimations)
        
        axis_y = QBarCategoryAxis()
        axis_y.append(categories)
        axis_y.setLabelsFont(self.chart_label_font)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)
        
        axis_x = QValueAxis()
        axis_x.setRange(0, 0.25)
        axis_x.setLabelsFont(self.chart_label_font)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        
        chart.legend().setVisible(False)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        
        chart_view.setMinimumHeight(400) 
        chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return chart_view

    def create_distribution_charts(self):
        np.random.seed(42)
        original_data = np.random.lognormal(mean=1.0, sigma=2.0, size=1000)
        transformed_data = np.log1p(original_data)
        
        # --- Chart 1: Original ---
        frequencies, bin_edges = np.histogram(original_data, bins=50)
        original_set = QBarSet("Original")
        original_set.append(list(frequencies))
        original_set.setColor(QColor("#f43f5e")) # Rose
        original_series = QBarSeries()
        original_series.append(original_set)
        chart_original = QChart()
        chart_original.addSeries(original_series)
        chart_original.setTitle("Histogram: Capital Gain (Original)")
        chart_original.legend().setVisible(False)
        chart_original.createDefaultAxes()
        chart_original.setTitleFont(self.chart_title_font)
        chart_original.axisX().setLabelsFont(self.chart_label_font)
        chart_original.axisY().setLabelsFont(self.chart_label_font)
        
        chart_original.setAnimationOptions(QChart.AllAnimations)
        
        chart_view_original = QChartView(chart_original)
        chart_view_original.setRenderHint(QPainter.Antialiasing)

        # --- Chart 2: Log-Transformed ---
        log_freq, log_bin_edges = np.histogram(transformed_data, bins=50)
        log_set = QBarSet("Log-Transformed")
        log_set.append(list(log_freq))
        log_set.setColor(QColor("#22c55e")) # Green
        log_series = QBarSeries()
        log_series.append(log_set)
        chart_log = QChart()
        chart_log.addSeries(log_series)
        chart_log.setTitle("Histogram: Capital Gain (Log-Transformed)")
        chart_log.legend().setVisible(False)
        chart_log.createDefaultAxes()
        chart_log.setTitleFont(self.chart_title_font)
        chart_log.axisX().setLabelsFont(self.chart_label_font)
        chart_log.axisY().setLabelsFont(self.chart_label_font)
        
        chart_log.setAnimationOptions(QChart.AllAnimations)
        
        chart_view_log = QChartView(chart_log)
        chart_view_log.setRenderHint(QPainter.Antialiasing)
        
        container_widget = QWidget()
        layout = QHBoxLayout(container_widget)
        layout.addWidget(chart_view_original)
        layout.addWidget(chart_view_log)
        
        container_widget.setMinimumHeight(350)
        container_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return container_widget

    def create_roc_curve_chart(self):
        roc_series = QSplineSeries()
        roc_series.setName("ROC Curve (AUC = 0.8945)")
        roc_data = [(0.0, 0.0), (0.01, 0.2), (0.05, 0.5), (0.1, 0.7), (0.15, 0.78),
                    (0.2, 0.82), (0.3, 0.88), (0.4, 0.92), (0.5, 0.95), (0.6, 0.97),
                    (0.7, 0.98), (0.8, 0.99), (0.9, 0.995), (1.0, 1.0)]
        for p in roc_data: roc_series.append(QPointF(p[0], p[1]))
        
        pen_roc = roc_series.pen()
        pen_roc.setColor(QColor("#0ea5e9")) # Sky Blue
        pen_roc.setWidth(3)
        roc_series.setPen(pen_roc)
        
        guess_series = QLineSeries()
        guess_series.setName("Random Guess")
        guess_series.append(QPointF(0.0, 0.0))
        guess_series.append(QPointF(1.0, 1.0))
        pen_guess = guess_series.pen()
        pen_guess.setStyle(Qt.DashLine)
        pen_guess.setColor(QColor("#f43f5e")) # Rose
        pen_guess.setWidth(2)
        guess_series.setPen(pen_guess)
        
        chart = QChart()
        chart.addSeries(roc_series)
        chart.addSeries(guess_series)
        chart.setTitle("Receiver Operating Characteristic (ROC) Curve")
        chart.createDefaultAxes()
        chart.axisX().setRange(0, 1)
        chart.axisY().setRange(0, 1)
        chart.legend().setAlignment(Qt.AlignRight)
        
        chart.setAnimationOptions(QChart.AllAnimations)
        
        chart.setTitleFont(self.chart_title_font)
        chart.legend().setFont(self.chart_label_font)
        chart.axisX().setTitleText("False Positive Rate (FPR)")
        chart.axisX().setTitleFont(self.chart_axis_title_font)
        chart.axisX().setLabelsFont(self.chart_label_font)
        chart.axisY().setTitleText("True Positive Rate (TPR)")
        chart.axisY().setTitleFont(self.chart_axis_title_font)
        chart.axisY().setLabelsFont(self.chart_label_font)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        
        chart_view.setMinimumHeight(350)
        chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return chart_view

    def create_confusion_matrix(self):
        tn_val, fn_val = "6215", "862"
        matrix_container = QFrame()
        layout = QGridLayout(matrix_container)
        layout.setSpacing(0)
        layout.addWidget(QLabel("<b>True Label</b>"), 1, 0, 2, 1, Qt.AlignCenter)
        layout.addWidget(QLabel("<b>Predicted Label</b>"), 0, 1, 1, 2, Qt.AlignCenter)
        layout.addWidget(QLabel("<=50K"), 1, 1, Qt.AlignCenter)
        layout.addWidget(QLabel(">50K"), 1, 2, Qt.AlignCenter)
        layout.addWidget(QLabel("<=50K"), 2, 0, Qt.AlignCenter)
        layout.addWidget(QLabel(">50K"), 3, 0, Qt.AlignCenter)
        tn_cell = self.create_matrix_cell(tn_val, "#1f2937", "#FFFFFF")
        fp_cell = self.create_matrix_cell("450", "#1f2937", "#FFFFFF")
        fn_cell = self.create_matrix_cell(fn_val, "#dbeafe", "#1f2937")
        tp_cell = self.create_matrix_cell("2800", "#dbeafe", "#1f2937")
        layout.addWidget(tn_cell, 2, 1)
        layout.addWidget(fp_cell, 2, 2)
        layout.addWidget(fn_cell, 3, 1)
        layout.addWidget(tp_cell, 3, 2)
        main_v_layout = QVBoxLayout()
        main_v_layout.addWidget(QLabel("Model Confusion Matrix", objectName="cardTitle"), 0, Qt.AlignCenter)
        main_v_layout.addWidget(matrix_container)
        final_widget = QWidget()
        final_widget.setLayout(main_v_layout)
        return final_widget

    def create_matrix_cell(self, text, bg_color, text_color):
        cell = QLabel(text)
        cell.setAlignment(Qt.AlignCenter)
        cell.setMinimumSize(100, 100)
        cell.setFont(QFont("Arial", 18, QFont.Bold))
        cell.setStyleSheet(f"background-color: {bg_color}; color: {text_color}; border: 1px solid #9CA3AF;")
        return cell

    def create_income_by_education_chart(self):
        series = QHorizontalBarSeries()
        data = [("Doctorate", 60), ("Bachelors", 50), ("Assoc-voc", 30), 
                ("HS-grad", 25), ("10th", 15), ("9th", 10), ("Preschool", 5)]
        bar_set = QBarSet("High Income %")
        categories = []
        for item in reversed(data):
            bar_set.append(item[1])
            categories.append(item[0])
            
        bar_set.setColor(QColor("#0d9488")) # Teal
        series.append(bar_set)
        
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("High Income % by Education")
        
        chart.setAnimationOptions(QChart.AllAnimations)
        
        chart.setTitleFont(self.chart_title_font)
        axis_y = QBarCategoryAxis()
        axis_y.append(categories)
        axis_y.setLabelsFont(self.chart_label_font)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)
        axis_x = QValueAxis()
        axis_x.setLabelsFont(self.chart_label_font)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        chart.legend().setVisible(False)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        
        chart_view.setMinimumHeight(350)
        chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return chart_view

    def create_income_by_gender_chart(self):
        set_male = QBarSet("Male")
        set_female = QBarSet("Female")
        set_male.append([20000, 9000])
        set_female.append([12000, 1500])
        
        set_male.setColor(QColor("#0ea5e9")) # Sky Blue
        set_female.setColor(QColor("#db2777")) # Pink
        
        series = QBarSeries()
        series.append(set_male)
        series.append(set_female)
        
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Income Distribution by Gender")
        chart.legend().setAlignment(Qt.AlignRight)
        
        chart.setAnimationOptions(QChart.AllAnimations)
        
        chart.setTitleFont(self.chart_title_font)
        chart.legend().setFont(self.chart_label_font)

        categories = ["<=50K", ">50K"]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        axis_x.setLabelsFont(self.chart_label_font)
        chart.addAxis(axis_x, Qt.AlignBottom)
        series.attachAxis(axis_x)
        
        axis_y = QValueAxis()
        axis_y.setLabelsFont(self.chart_label_font)
        chart.addAxis(axis_y, Qt.AlignLeft)
        series.attachAxis(axis_y)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        
        chart_view.setMinimumHeight(350)
        chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return chart_view

    def create_income_trends_chart(self):
        series = QSplineSeries()
        data = [(10,1), (15,2), (20,3), (25,8), (30,12), (35,20), (40,30), (45,38), 
                (50,42), (55,40), (60,35), (65,30), (70,28), (75,22), (80,30), 
                (85,25), (90,30)]
        for point in data: series.append(QPointF(point[0], point[1]))
        
        pen = series.pen()
        pen.setColor(QColor("#0ea5e9")) # Sky Blue
        pen.setWidth(3)
        series.setPen(pen)
        
        chart = QChart()
        chart.addSeries(series)
        chart.createDefaultAxes()
        chart.legend().setVisible(False)
        
        chart.setAnimationOptions(QChart.AllAnimations)
        
        chart.setTitleFont(self.chart_title_font)
        chart.axisX().setTitleText("Age")
        chart.axisX().setTitleFont(self.chart_axis_title_font)
        chart.axisX().setLabelsFont(self.chart_label_font)
        chart.axisY().setTitleText("High Income Percentage (%)")
        chart.axisY().setTitleFont(self.chart_axis_title_font)
        chart.axisY().setLabelsFont(self.chart_label_font)
        
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        
        chart_view.setMinimumHeight(350) 
        chart_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return chart_view
