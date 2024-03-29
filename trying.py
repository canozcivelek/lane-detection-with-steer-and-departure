import cv2
import numpy as np
import math
def detect_lanes(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Define region of interest (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Perform Hough Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 15, minLineLength=40, maxLineGap=20)
    
    return lines

def draw_lanes(image, lines):
    lane_image = np.zeros_like(image)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    return lane_image

import math

def calculate_max_velocity(curvature, friction_coefficient=0.7):
    # Coefficient of friction between tires and road surface
    mu = friction_coefficient

    # Acceleration due to gravity (m/s^2)
    g = 9.81

    # Calculate maximum safe velocity based on curvature
    max_velocity = math.sqrt(mu * g * curvature)
    
    return max_velocity


def calculate_curvature(lines):
    # Separate left and right lane lines
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:
            left_lines.append(line)
        elif slope > 0:
            right_lines.append(line)

    # Fit polynomial to left and right lane lines
    left_points = np.array([[x1, y1] for line in left_lines for x1, y1, x2, y2 in line])
    right_points = np.array([[x1, y1] for line in right_lines for x1, y1, x2, y2 in line])

    if len(left_points) == 0 or len(right_points) == 0:
        return 0  # Curvature calculation not possible

    left_coefficients = np.polyfit(left_points[:, 0], left_points[:, 1], 2)
    right_coefficients = np.polyfit(right_points[:, 0], right_points[:, 1], 2)

    # Define y-value where we want radius of curvature
    plot_y = 720  # Assuming height of the frame is 720 pixels

    # Calculate radii of curvature
    left_curvature = ((1 + (2 * left_coefficients[0] * plot_y + left_coefficients[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_coefficients[0])
    right_curvature = ((1 + (2 * right_coefficients[0] * plot_y + right_coefficients[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_coefficients[0])

    # Average curvature of left and right lanes
    curvature = (left_curvature + right_curvature) / 2

    return curvature


def mark_lane_area(image, lines):
    lane_image = np.zeros_like(image)
    
    if lines is not None:
        left_points = []
        right_points = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_points.append((x1, y1))
                left_points.append((x2, y2))
            elif slope > 0:
                right_points.append((x1, y1))
                right_points.append((x2, y2))
        
        # Convert points to numpy arrays
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        
        # Draw filled polygons for the lane area
        cv2.fillPoly(lane_image, [left_points], color=(0, 255, 0))
        cv2.fillPoly(lane_image, [right_points], color=(0, 255, 0))
    
    return lane_image

def main():
    # Load video
    cap = cv2.VideoCapture('drive.mp4')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect lanes
        lines = detect_lanes(frame)
        
        # Mark lane area on the original frame
        lane_area = mark_lane_area(frame, lines)
        
        # Calculate curvature
        curvature = calculate_curvature(lines)
        
        # Calculate maximum safe velocity
        max_velocity = calculate_max_velocity(curvature)
        
        # Display information on the frame
        cv2.putText(frame, f"Curvature: {curvature:.2f} m", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Max Velocity: {max_velocity:.2f} m/s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the original frame with detected lanes and marked lane area
        cv2.imshow('Lane Detection', np.hstack((frame, lane_area)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
