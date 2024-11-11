import cv2
import numpy as np

# Open the image
image = cv2.imread("./s-l1600.webp")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find the edges in the image
edges = cv2.Canny(gray, 200, 300)

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Initialize variables to store extreme values
height, width = image.shape[:2]
lowest_x = width
highest_x = 0
lowest_y = height
highest_y = 0

# Loop through each line to find extreme points
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Update leftmost (lowest_x) and rightmost (highest_x) points
        lowest_x = min(lowest_x, x1, x2)
        highest_x = max(highest_x, x1, x2)
        
        # Update topmost (lowest_y) and bottommost (highest_y) points
        lowest_y = min(lowest_y, y1, y2)
        highest_y = max(highest_y, y1, y2)

# Draw a rectangle around the detected card area based on extreme points
cv2.rectangle(image, (lowest_x, lowest_y), (highest_x, highest_y), (0, 255, 0), 2)

# Display the result
cv2.imshow("Detected Card Boundaries", cv2.resize(image, (0, 0), fx=0.5, fy=0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()




#take only the cropped image
cropped_image = image[lowest_y:highest_y, lowest_x:highest_x]

# Display the cropped image

cv2.imshow("Cropped Image", cv2.resize(cropped_image, (0, 0), fx=0.5, fy=0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()







