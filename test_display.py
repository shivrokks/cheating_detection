#!/usr/bin/env python3
import cv2
import numpy as np

# Create a simple test image
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(test_image, "Test Display", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

print("Attempting to display test window...")
cv2.imshow("Test Window", test_image)
print("Test window created, waiting for key press...")
cv2.waitKey(3000)  # Wait 3 seconds
cv2.destroyAllWindows()
print("Test completed")
