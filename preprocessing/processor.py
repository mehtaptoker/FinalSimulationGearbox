import cv2
import json
import numpy as np
import os

class Processor:
    @staticmethod
    def process_input(png_path, constraints_path, output_path, config=None):
        """
        Process input files and generate intermediate representation
        
        Args:
            png_path: Path to input PNG image
            constraints_path: Path to JSON constraints file
            output_path: Path for output JSON file
            config: Optional configuration dictionary
            
        Returns:
            Loaded constraints object
        """
        # Load constraints from JSON
        with open(constraints_path, 'r') as f:
            constraints = json.load(f)
        
        # Load and process image
        img = cv2.imread(png_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {png_path}")
        
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Detect red (input) shaft
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        input_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        input_center = Processor._get_contour_center(max(input_contours, key=cv2.contourArea)) if input_contours else None
        
        # Detect green (output) shaft
        lower_green = np.array([35, 120, 70])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        output_contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_center = Processor._get_contour_center(max(output_contours, key=cv2.contourArea)) if output_contours else None
        
        # Detect and approximate boundary using edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        boundary_contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary = Processor._approximate_contour(max(boundary_contours, key=cv2.contourArea)) if boundary_contours else []
        
        # Create and save intermediate representation
        intermediate = {
            "boundaries": boundary,
            "input_shaft": {"x": int(input_center[0]), "y": int(input_center[1])} if input_center else None,
            "output_shaft": {"x": int(output_center[0]), "y": int(output_center[1])} if output_center else None
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(intermediate, f, indent=2)
        
        return constraints

    @staticmethod
    def _get_contour_center(contour):
        """Calculate center of mass for a contour"""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return (0, 0)
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        return (cX, cY)

    @staticmethod
    def _approximate_contour(contour, epsilon_factor=0.01):
        """Approximate contour using Ramer-Douglas-Peucker algorithm"""
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True).squeeze().tolist()
