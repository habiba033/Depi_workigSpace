import sys
import numpy as np
from PyQt5 import uic

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton,QGraphicsScene, QVBoxLayout, QFileDialog,QGraphicsView
from PyQt5.QtGui import QPixmap  # Correct import for QPixmap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class ImageClassificationAPP(QWidget):
    
    def __init__(self):
        super(ImageClassificationAPP, self).__init__()
        uic.loadUi(r'D:/InnovionRay_team/Innovisionray_team/InnoVisionRay/Amit/AI_Diploma/AI Engineering/Computer_Vision/CNN_Project/CNN_project_qt5/GUI Detection.ui', self)

        # Load buttons
        self.load_button_1 = self.findChild(QPushButton, 'pushButton')
        self.load_button_1.clicked.connect(self.load_image)

        self.load_button_2 = self.findChild(QPushButton, 'pushButton_2')
        self.load_button_2.clicked.connect(self.predict_with_model)

        # Image viewer
        self.image_viewer_1 = self.findChild(QGraphicsView, 'graphicsView')
        self.scene = QGraphicsScene(self)
        self.image_viewer_1.setScene(self.scene)

        # Result labels
        self.result_label = self.findChild(QLabel, 'label')
        self.result_label_2 = self.findChild(QLabel, 'label_2')

        # Load model
        self.model = load_model(r'D:/InnovionRay_team/Innovisionray_team/InnoVisionRay/Amit/AI_Diploma/AI Engineering/Computer_Vision/CNN_Project/CNN_project_qt5/output/best_model.h5')

        # Class names mapping
        self.classes = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

        self.image_path = None

    
    def load_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg);;All Files (*)", options=options)
        if self.image_path:
            try:
                print("Image path:", self.image_path)  # للتأكد من المسار
                
                img = cv2.imread(self.image_path)
                if img is None:
                    raise ValueError(f"Image not found or unsupported format: {self.image_path}")

                # Resize proportionally
                h, w = img.shape[:2]
                scale = min(700 / w, 550 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # BGR -> RGB
                height, width, _ = img_resized.shape
                bytes_per_line = 3 * width
                q_img = QImage(img_resized.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

                # Show in scene
                self.scene.clear()
                self.scene.addPixmap(QPixmap.fromImage(q_img))
                self.image_viewer_1.setScene(self.scene)

            except Exception as e:
                self.result_label.setText(f"Error loading image: {e}")


    def preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (100, 100))  # Resize to model input size
            img = img.astype('float32') / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            return img
        except Exception as e:
            self.result_label.setText(f"Error processing image: {e}")
            return None

    def predict_with_model(self):
        if self.image_path:
            img = self.preprocess_image(self.image_path)
            if img is not None:
                try:
                    predictions = self.model.predict(img)
                    class_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100

                    # Update labels with class names and confidence
                    self.result_label.setText(f'Class: {self.classes[class_idx]}')
                    self.result_label_2.setText(f'Confidence: {confidence:.2f}%')
                except Exception as e:
                    self.result_label.setText(f"Prediction error: {e}")
        else:
            self.result_label.setText("Please load an image first.")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    classifier = ImageClassificationAPP()
    classifier.show()
    sys.exit(app.exec_())