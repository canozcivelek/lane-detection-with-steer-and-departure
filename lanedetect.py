# import cv2
# import numpy as np

# def region_of_interest(img, vertices):
#     mask = np.zeros_like(img)
#     cv2.fillPoly(mask, vertices, 255)
#     masked_img = cv2.bitwise_and(img, mask)
#     return masked_img

# def draw_lines(img, lines, color=[0, 255, 255], thickness=3, y_min=0):
#     for line in lines:
#         if line is not None:
#             for x1, y1, x2, y2 in line:
#                 if y1 >= y_min and y2 >= y_min:  # Check if both points are below the minimum y-coordinate
#                     cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# def detect_lanes(img):
#     # Convert image to grayscale
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur
#     blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
#     # Apply Canny edge detection
#     edges = cv2.Canny(blur_img, 50, 150)
    
#     # Define region of interest
#     height, width = img.shape[:2]
#     roi_top = int(height * 0.50)  # Adjust the ratio as per your requirement
#     vertices = np.array([[(0, height), (width/2, roi_top), (width, height)]], dtype=np.int32)
#     masked_edges = region_of_interest(edges, vertices)
    
#     # Perform Hough transform to detect lines
#     lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)
    
#     # Filter and group lines to form a single continuous line for each side
#     left_lines, right_lines = group_lines(lines, roi_top)
    
#     # Draw lines on the original image
#     lane_img = np.copy(img) * 0  # Creating a blank to draw lines on
#     draw_lines(lane_img, [left_lines, right_lines], y_min=roi_top)
    
#     return lane_img

# def group_lines(lines, y_min):
#     left_lines = []
#     right_lines = []
#     if lines is not None:
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 slope = (y2 - y1) / (x2 - x1)
#                 if abs(slope) > 0.5:  # Filter out lines with slope close to horizontal
#                     if slope < 0 and y1 >= y_min and y2 >= y_min:  # Left lane
#                         left_lines.append(line)
#                     elif slope > 0 and y1 >= y_min and y2 >= y_min:  # Right lane
#                         right_lines.append(line)
#     # Average the lines to get a single line for each side
#     left_avg_line = np.mean(left_lines, axis=0, dtype=np.int32) if left_lines else None
#     right_avg_line = np.mean(right_lines, axis=0, dtype=np.int32) if right_lines else None
#     return left_avg_line, right_avg_line

# # Open video file
# cap = cv2.VideoCapture('drive.mp4')

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Detect lanes in the frame
#     lane_img = detect_lanes(frame)
    
#     # Combine original frame with detected lanes
#     result = cv2.addWeighted(frame, 1, lane_img, 0.5, 0)
    
#     # Display the frame with detected lanes
#     cv2.imshow('Lanes', result)
    
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# # Release video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()




# import cv2
# import numpy as np

# def region_of_interest(img, vertices):
#     mask = np.zeros_like(img)
#     cv2.fillPoly(mask, vertices, 255)
#     masked_img = cv2.bitwise_and(img, mask)
#     return masked_img

# def draw_lines(img, curves, color=[0, 255, 255], thickness=40, y_min=0):
#     for curve in curves:
#         if curve is not None and len(curve) > 0:
#             curve = curve.reshape((-1, 1, 2))
#             cv2.polylines(img, [curve], False, color, thickness)

# def detect_lanes(img):
#     # Convert image to grayscale
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur
#     blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
#     # Apply Canny edge detection
#     edges = cv2.Canny(blur_img, 50, 150)
    
#     # Define region of interest
#     height, width = img.shape[:2]
#     roi_top = int(height * 0.6)  # Adjust the ratio as per your requirement
#     vertices = np.array([[(0, height), (width/2, roi_top), (width, height)]], dtype=np.int32)
#     masked_edges = region_of_interest(edges, vertices)
    
#     # Perform Hough transform to detect lines
#     lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=0, maxLineGap=300)
    
#     # Fit curves to the detected lines
#     left_curve = fit_curve(lines, roi_top, width, direction='left')
#     right_curve = fit_curve(lines, roi_top, width, direction='right')
    
#     # Draw curves on the original image
#     lane_img = np.copy(img) * 0  # Creating a blank to draw curves on
#     draw_lines(lane_img, [left_curve, right_curve], color=[0, 0, 255], thickness=5, y_min=roi_top)
    
#     return lane_img

# def fit_curve(lines, y_min, img_width, direction='left'):
#     points = []
#     if lines is not None:
#         for line in lines:
#             for x1, y1, x2, y2 in line:
#                 slope = (y2 - y1) / (x2 - x1)
#                 if abs(slope) > 0.5:  # Filter out lines with slope close to horizontal
#                     if direction == 'left' and slope < 0 and y1 >= y_min and y2 >= y_min:  # Left curve
#                         points.extend([(x1, y1), (x2, y2)])
#                     elif direction == 'right' and slope > 0 and y1 >= y_min and y2 >= y_min:  # Right curve
#                         points.extend([(x1, y1), (x2, y2)])
    
#     if points:
#         points = np.array(points)
#         curve = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)  # Fit a curve using cv2.fitLine
#         curve = extend_line(curve, y_min, img_width)
#         return curve
#     else:
#         return None

# def extend_line(line, y_min, img_width):
#     vx, vy, x, y = line.squeeze()
#     slope = vy / vx
#     x1 = int(((y_min - y) / slope) + x)
#     y1 = y_min
#     x2 = int(((img_width - y) / slope) + x)
#     y2 = img_width
#     return np.array([(x1, y1), (x2, y2)])

# # Open video file
# cap = cv2.VideoCapture('drive.mp4')

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Detect lanes in the frame
#     lane_img = detect_lanes(frame)
    
#     # Combine original frame with detected lanes
#     result = cv2.addWeighted(frame, 1, lane_img, 0.5, 0)
    
#     # Display the frame with detected lanes
#     cv2.imshow('Lanes', result)
    
#     if cv2.waitKey(15) & 0xFF == ord('q'):
#         break

# # Release video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_curves(img, curves, color=[0, 255, 255], thickness=40, y_min=0):
    for curve in curves:
        if curve is not None and len(curve) > 0:
            curve = curve.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [curve], False, color, thickness)

def detect_lanes(img):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blur_img, 50, 150)
    
    # Define region of interest
    height, width = img.shape[:2]
    roi_top = int(height * 0.59)  # Adjust the ratio as per your requirement
    vertices = np.array([[(0, height), (width/2, roi_top), (width, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    # Perform Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=100, maxLineGap=100)
    
    # Fit curves to the relevant detected lines
    left_curve = fit_curve(lines, roi_top, width, direction='left')
    right_curve = fit_curve(lines, roi_top, width, direction='right')
    
    # Draw curves on the original image
    lane_img = np.copy(img) * 0  # Creating a blank to draw curves on
    draw_curves(lane_img, [left_curve, right_curve], color=[0, 0, 255], thickness=5, y_min=roi_top)
    
    return lane_img

def fit_curve(lines, y_min, img_width, direction='left'):
    points = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 0.3 and abs(slope) < 3:  # Filter out lines with slope close to horizontal or vertical
                    if direction == 'left' and slope < 0 and y1 >= y_min and y2 >= y_min:  # Left curve
                        points.extend([(x1, y1), (x2, y2)])
                    elif direction == 'right' and slope > 0 and y1 >= y_min and y2 >= y_min:  # Right curve
                        points.extend([(x1, y1), (x2, y2)])
    
    if points:
        points = np.array(points)
        curve = interpolate_curve(points)
        return curve
    else:
        return None

def interpolate_curve(points):
    x = points[:, 0]
    y = points[:, 1]
    
    # Compute weights based on y-values
    weights = np.sqrt(np.abs(y - np.mean(y)))
    
    # Fit a polynomial of degree 2 (quadratic) with weighted least squares
    coeffs = np.polyfit(y, x, 2, w=weights)
    
    interpolated_y = np.linspace(min(y), max(y), 100)
    interpolated_x = np.polyval(coeffs, interpolated_y)
    interpolated_curve = np.column_stack((interpolated_x, interpolated_y))
    
    return interpolated_curve

# Open video file
cap = cv2.VideoCapture('drive.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect lanes in the frame
    lane_img = detect_lanes(frame)
    
    # Combine original frame with detected lanes
    result = cv2.addWeighted(frame, 1, lane_img, 0.5, 0)
    
    # Display the frame with detected lanes
    cv2.imshow('Lanes', result)
    
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
