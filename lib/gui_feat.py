import numpy as np
import cv2
import sys
import qtpy
from qtpy.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox, QPushButton
from qtpy.QtGui import QPixmap, QImage
from qtpy.QtCore import Qt

def qimage2numpy(qimage: QPixmap) -> np.ndarray:
    """Convert QImage to numpy array.

    Args:
        qimage (QPixmap): QPixmap to be converted.

    Returns:
        np.ndarray: Converted numpy array.
    """
    image = qimage.toImage()
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    np_image = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return np_image

def resize_aspect_ratio(img:np.ndarray, max_width:int, max_height:int):
    """
    Resize an image while maintaining its aspect ratio.
    Args:
        img (np.ndarray): The image to be resized.
        max_width (int): The maximum width of the resized image.
        max_height (int): The maximum height of the resized image.
    Returns:
        np.ndarray: The resized image.
    
    Usage:
    image = cv2.imread("path_to_your_image.jpg")  # Replace with the path to your image
    resized_image = resize_aspect_ratio(image, max_width=500, max_height=500)
    """
    height, width = img.shape[:2]
    aspect_ratio = width / height
    if width > height:
        # Width is the limiting factor
        new_width = min(width, max_width)
        new_height = int(new_width / aspect_ratio)
    else:
        # Height is the limiting factor
        new_height = min(height, max_height)
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_img
    

class ImageViewer(QWidget):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        
        # The following variables is for the original image (as is)
        self.original_image = self.convert_cv_qt(image)
        self.processed_image = self.original_image.copy()
        
        # そのままでは画像が大きいと表示できないのでアスペクト比を保ったまま縮小する可視化用の画像の変数を作成する
        
        # Layouts
        self.main_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        self.params_layout = QVBoxLayout()
        
        # image display
        scaled_original_image = self.convert_cv_qt(resize_aspect_ratio(image, max_width=500, max_height=500))
        scaled_processed_image = self.convert_cv_qt(resize_aspect_ratio(image, max_width=500, max_height=500))
        self.original_label = QLabel()
        self.original_label.setPixmap(scaled_original_image)
        self.processed_label = QLabel()
        self.processed_label.setPixmap(scaled_processed_image)
        # method selection
        self.method_combo = QComboBox()
        self.method_combo.addItems(["None", "Gaussian Blur", "Sobel"])
        self.method_combo.currentIndexChanged.connect(self.update_method)
        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_filter)
        # set up layouts
        self.right_layout.addWidget(self.method_combo)
        self.right_layout.addLayout(self.params_layout)
        self.right_layout.addWidget(self.apply_button)
        self.main_layout.addWidget(self.original_label)
        self.main_layout.addLayout(self.right_layout)
        self.main_layout.addWidget(self.processed_label)
        
        self.setLayout(self.main_layout)
        self.show()

    def convert_cv_qt(self, cv_img):
        """ Convert an OpenCV image to QPixmap """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)
    
    def resize_event(self, event):
        # Resize and update the images when the window is resized
        self.resize_image_label(self.original_label, self.original_image_cv)
        self.resize_image_label(self.processed_label, self.processed_image_cv)
        super().resize_event(event)
        
    def resize_image_label(self, label, cv_image):
        # Get the new size from the window
        new_width = label.width()
        new_height = label.height()
        # Convert and resize the image
        resized_image = cv2.resize(cv_image, (new_width, new_height))
        qt_image = self.convert_cv_qt(resized_image)
        # Update the label pixmap
        label.setPixmap(qt_image)
    
    def update_method(self):
        for i in range(self.params_layout.count()):
            self.params_layout.itemAt(i).widget().setParent(None)
        if self.method_combo.currentText() == "Gaussian Blur":
            self.param1 = QSpinBox()
            self.params_layout.addWidget(self.param1)
        elif self.method_combo.currentText() == "Sobel":
            self.param1 = QSpinBox()
            self.params_layout.addWidget(self.param1)
    
    def apply_filter(self):
        image = qimage2numpy(self.original_label.pixmap())
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)  # Convert from RGBA to BGR

        if self.method_combo.currentText() == "Gaussian Blur":
            kernel_size = self.param1.value() | 1  # Ensure the kernel size is odd
            processed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif self.method_combo.currentText() == "Sobel":
            kernel_size = self.param1.value() | 1  # Ensure the kernel size is odd
            processed_image = cv2.Sobel(image, cv2.CV_8U, 1, 1, ksize=kernel_size)
        else:
            processed_image = image  # No processing

        self.processed_image = self.convert_cv_qt(processed_image)
        self.processed_label.setPixmap(self.processed_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    image = cv2.imread("data/brandenburg_gate/images/00883281_9633489441.jpg")
    if image is not None:
        gui = ImageViewer(image)
    else:
        print("Image not found.")
    sys.exit(app.exec_())