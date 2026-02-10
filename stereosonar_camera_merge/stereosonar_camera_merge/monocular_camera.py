# Python libraries
import numpy as np
from ultralytics import YOLO

# OpenCV
import cv2

cv2.setNumThreads(4)  # Adjust the number based on your CPU cores
cv2.setUseOptimized(True)

class MonocularCamera:
    """Class to handle operations related to a monocular camera, including preprocessing 
    and segmentation of images.

    Attributes:
        K (numpy.ndarray): The optimized intrinsic camera matrix.
        height (int): The height of the RGB image.
        width (int): The width of the RGB image.
        yolo_model (yolo model): Trained yolo model
    """

    def __init__(self, K, D, rgb_width, rgb_height, model_path):
        """
        Initializes the MonocularCamera object.
        """
        self.K, roi = cv2.getOptimalNewCameraMatrix(K, D, (rgb_width, rgb_height), 1, (rgb_width,rgb_height))
        self.height = rgb_height
        self.width = rgb_width
        self.yolo_model = YOLO(model_path)


    def yolo_segment(self, image):
        confidences = None
        height, width = image.shape[:2]
        labeled_image = np.zeros(image.shape[:2], dtype=np.uint8)
        confidence_image = np.zeros(image.shape[:2], dtype=np.uint8)
        labels = np.array([1])
        seg_results = self.yolo_model(image, verbose=False,  conf = 0.25)# confidence level needed to determine positive sample
        r = seg_results[0]

        if r.masks is not None:
            # Prepare masks and initialize output
            masks = r.masks.data.cpu().numpy()  # Shape: (N, H, W)
            boxes = r.boxes                              # YOLOv8 bounding boxes Shape: (N, )

            # Generate labels
            labels = np.arange(masks.shape[0]+2)  # Avoid using 0 (background) and 1 (reserved)

            # Apply labels to the image
            for i, (mask) in enumerate(masks):
                # Resize mask to match original image size
                resized_mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                labeled_image[resized_mask.astype(bool)] = labels[i+2]

        return labels, labeled_image
