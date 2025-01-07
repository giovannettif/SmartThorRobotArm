from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

def main():
    # Load the depth estimation model
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    cap = cv2.VideoCapture(1)  # Use the birds eye view camera.

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        # Resize the frame to a smaller size (e.g., 640x480 or 320x240)
        frame_resized = cv2.resize(frame, (160, 120))  # or any resolution that balances speed and accuracy
        image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        output_depth = pipe(image)["depth"]
        depth_image = np.array(output_depth)

        heatmap = cv2.applyColorMap(np.uint8(output_depth), cv2.COLORMAP_JET)
        output_image = cv2.addWeighted(frame_resized, 1, heatmap, 1, 0)

        # Show the original frame and the output image with heatmap overlay
        # cv2.imshow("Original", frame)
        cv2.imshow("Heatmap Overlay", output_image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()