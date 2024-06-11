import cv2

# Initialize the list to store points
points = []


# Mouse callback function to capture click events
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Draw a circle at the clicked point
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        # Show the image with the annotated point
        cv2.imshow("image", img)
        print(f"Point recorded: ({x}, {y})")

        # Ask for a label for the clicked point
        label = input("Enter a label for this point: ")
        points.append({"coordinates": [x, y], "label": label})


# Load an image
img = cv2.imread("3D-Reconstruction/room_annotation/room.png")

# Create a window to display the image
cv2.imshow("image", img)

# Set the mouse callback function to capture clicks
cv2.setMouseCallback("image", click_event)

# Wait until the 'q' key is pressed to exit
print("Click on the image to annotate points. Press 'q' to exit.")
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Destroy all OpenCV windows
cv2.destroyAllWindows()

# Print the collected points
print("Annotated points:", points)

# Save the points to a file (optional)
import json

with open("3D-Reconstruction/room_annotation/annotated_points.json", "w") as f:
    json.dump(points, f, indent=4)

print("Annotated points saved to 'annotated_points.json'")
