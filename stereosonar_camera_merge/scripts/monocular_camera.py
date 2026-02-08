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
        kernel (numpy.ndarray): Structuring element used for morphological operations.
    """

    def __init__(self, K, D, rgb_width, rgb_height, model_path):
        """
        Initializes the MonocularCamera object.
        """
        self.K, roi = cv2.getOptimalNewCameraMatrix(K, D, (rgb_width, rgb_height), 1, (rgb_width,rgb_height))
        self.height = rgb_height
        self.width = rgb_width

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        self.yolo_model = YOLO(model_path)


    def preprocess(self, image, threshold_inv):
        """
        Preprocesses the input image by converting it to grayscale, applying Gaussian 
        blur, adaptive thresholding, and noise removal.

        Args:
            image (numpy.ndarray): The input color image (BGR format).
            threshold_inv (bool): If True, applies inverse adaptive thresholding (for dark areas).
                                  If False, applies regular adaptive thresholding (for light areas).

        Returns:
            numpy.ndarray: The processed binary image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        # Apply adaptive thresholding
        if threshold_inv:
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 501, 0)
        else:
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 0)

        # Noise removal using morphological opening
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, self.kernel, iterations=3)
        return gray
    

    def segment_image(self, image):
        """
        Segments the input binary image using morphological operations, distance 
        transformation, and the watershed algorithm.

        Args:
            image (numpy.ndarray): The preprocessed binary image.

        Returns:
            tuple:
                - numpy.ndarray: An array containing the unique labels of segmented regions.
                - numpy.ndarray: The labeled image after applying the watershed algorithm.
        """
        # Obtain sure background area
        sure_bg = cv2.dilate(image, self.kernel, iterations=2)

        # Compute the distance transform
        dist = cv2.distanceTransform(image, cv2.DIST_L2, 5)

        # Obtain sure foreground area
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        # Determine unknown region
        unknown_area = cv2.subtract(sure_bg, sure_fg)

        # Label connected components
        _, labeled_image = cv2.connectedComponents(sure_fg)

        # Adjust labels: Background becomes 1 instead of 0
        labeled_image += 1

        # Mark unknown region as 0
        labeled_image[unknown_area == 255] = 0

        # Convert grayscale image to color for watershed
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Apply watershed algorithm
        labeled_image = cv2.watershed(color_image, labeled_image)

        # Get unique labels
        labels = np.unique(labeled_image)

        return labels, labeled_image


    def yolo_segment(self, image):
        confidences = None
        height, width = image.shape[:2]
        labeled_image = np.zeros(image.shape[:2], dtype=np.uint8)
        confidence_image = np.zeros(image.shape[:2], dtype=np.uint8)
        labels = np.array([1])
        seg_results = self.yolo_model(image, verbose=False,  conf = 0.25)# confidence level needed to determine positive sample
        #seg_annotated = seg_result[0].plot(show=False)
        #self.image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="bgr8"))
        r = seg_results[0]

        
        if r.masks is not None:
            
            # Prepare masks and initialize output
            masks = r.masks.data.cpu().numpy()  # Shape: (N, H, W)
            #labeled_image = np.zeros(masks.shape[1:], dtype=np.uint8)

            boxes = r.boxes                              # YOLOv8 bounding boxes Shape: (N, )
            # confidences = boxes.conf.cpu().numpy()       # Confidence for each mask/detection
            # class_ids = boxes.cls.cpu().numpy()          # Class ID for each detection

            #for i, (mask, conf, cls_id) in enumerate(zip(masks, confidences, class_ids)):
            #    print(f"Object {i}: Class={int(cls_id)}, Confidence={conf:.2%}")

            # Generate labels
            labels = np.arange(masks.shape[0]+2)  # Avoid using 0 (background) and 1 (reserved)
            #print(labels)
            # Apply labels to the image
            #for i, (mask, conf) in enumerate(zip(masks, confidences)):
            for i, (mask) in enumerate(masks):
                #labeled_image[mask.astype(bool)] = labels[i+2]
                # Resize mask to match original image size
                resized_mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                labeled_image[resized_mask.astype(bool)] = labels[i+2]
                #confidence_image[resized_mask.astype(bool)] = confidences[i]*100

        return labels, labeled_image# , confidence_image, confidences
