
import cv2
import numpy as np
import os

# Define variables for meter-to-pixel conversion
ym_per_pix = 30 / 720
xm_per_pix = 3.7 / 720

# Get current working directory
CWD_PATH = os.getcwd()

# Function to read video
def readVideo():
    # Read input video from current working directory
    inpVideo = cv2.VideoCapture(os.path.join(CWD_PATH, 'drive1.mp4'))
    return inpVideo

# Function to process image
def processImage(inpImage):
    # Apply HLS color filtering to filter out white lane lines
    hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 160, 10])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(hls, lower_white, upper_white)
    hls_result = cv2.bitwise_and(inpImage, inpImage, mask=mask)

    # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    canny = cv2.Canny(blur, 40, 60)

    return canny

# Function to calculate curve radius
def measure_lane_curvature(ploty, left_fit, right_fit):
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fit * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fit * xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
    
    # Return the radius of curvature for both lanes
    return left_curverad, right_curverad

# Function to fit lane lines
def fit_lane_lines(binary_warped):
    # Perform sliding window search to detect lane pixels
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set parameters for sliding windows
    nwindows = 9
    margin = 100
    minpix = 50
    window_height = binary_warped.shape[0] // nwindows

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions for each window
    leftx_current = int(np.mean(nonzerox[nonzeroy == binary_warped.shape[0] - 1])) if (nonzerox[nonzeroy == binary_warped.shape[0] - 1]).any() else 0
    rightx_current = int(np.mean(nonzerox[nonzeroy == binary_warped.shape[0] - 1])) if (nonzerox[nonzeroy == binary_warped.shape[0] - 1]).any() else binary_warped.shape[1]

    # Lists to hold the left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify nonzero pixels within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


# Main function
def main():
    # Read the video
    inpVideo = readVideo()

    while inpVideo.isOpened():
        # Read a frame from the video
        ret, frame = inpVideo.read()

        if not ret:
            break

        # Process the frame
        processed_image = processImage(frame)

        # Fit lane lines
        left_fit, right_fit = fit_lane_lines(processed_image)

        # Generate x and y values for plotting
        ploty = np.linspace(0, processed_image.shape[0] - 1, processed_image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Calculate the radius of curvature
        left_curverad, right_curverad = measure_lane_curvature(ploty, left_fitx, right_fitx)
        curvature = (left_curverad + right_curverad) / 2

        # Draw the lane lines
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(processed_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space
        newwarp = cv2.warpPerspective(color_warp, M_inv, (frame.shape[1], frame.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)

        # Display curvature on the image
        cv2.putText(result, 'Radius of Curvature: {:.2f} m'.format(curvature), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Lane Detection', result)
        cv2.waitKey(1)

    inpVideo.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Define perspective transform matrix M and its inverse M_inv
    src = np.float32([(210, 720), (550, 470), (720, 470), (1110, 720)])  # Source points
    dst = np.float32([(320, 720), (320, 0), (960, 0), (960, 720)])        # Destination points
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    main()

