import torch
from PIL import Image
import numpy as np
import cv2
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import threading
import tkinter as tk
import NVIDIA_LLM_API as NIM
import moveArm
import time

# Load the CLIPSeg model and processor
processor = CLIPSegProcessor.from_pretrained("clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("clipseg-rd64-refined")

# Global variable for the prompt and threshold
current_prompt = "background"
current_threshold = 0.4

def pick_up_highlighter():
    time.sleep(3)
    moveArm.send_command(13, 80, 57, 0, 2, 6) # Pre-calculated for picking up highlighter
    time.sleep(8)
    moveArm.send_command(70, 80, 0, 0, 2, 6) # Pre-calculated for picking up highlighter

def pick_up_screwdriver():
    time.sleep(3)
    moveArm.send_command(-25, 70, 70, 0, -10, -6) # Pre-calculated for picking up highlighter
    time.sleep(8)
    moveArm.send_command(70, 80, 0, 0, 2, 6) # Pre-calculated for picking up highlighter

def process_frame(frame, prompt, threshold=0.4, alpha_value=0.5, draw_rectangles=True):
    # Prepare inputs
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits

    pred = torch.sigmoid(preds)
    mat = pred.squeeze().cpu().numpy()
    mask = Image.fromarray(np.uint8(mat * 255), "L")
    mask = mask.convert("RGB")
    mask = mask.resize(image.size)
    mask = np.array(mask)[:, :, 0]
    mask_max = mask.max()
    mask_min = mask.min()
    mask = (mask - mask_min) / (mask_max - mask_min)
    bmask = mask > threshold
    mask[mask < threshold] = 0

    # create heatmap mask
    heatmap = cv2.applyColorMap(np.uint8(mask * 255), cv2.COLORMAP_JET)

    # Blend heatmap with the original frame
    output_image = cv2.addWeighted(frame, 1, heatmap, alpha_value, 0)

    # Find contours for bounding boxes
    if draw_rectangles:
        contours, _ = cv2.findContours(bmask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 0:  # Only consider non-zero area contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Yellow box

    return mask, output_image

def update_prompt(new_prompt):
    global current_prompt
    response = NIM.process_input_with_api(new_prompt)
    obj, action, target_obj, classification = NIM.process_csv_output(response)
    current_prompt = obj
    if (obj == "highlighter"):
       pick_up_highlighter()
    elif (obj == "screwdriver"):
        pick_up_screwdriver()
    else:
        print("Object Unidentified:", obj, action, target_obj, classification)
    # Update the label text when prompt changes

def update_threshold(new_threshold):
    global current_threshold
    current_threshold = new_threshold
    # Update the label text when threshold changes
    # threshold_label.config(text=f"Current Threshold: {current_threshold:.2f}")

def main(alpha_value=0.5, draw_rectangles=True):
    cap = cv2.VideoCapture(1)  # Use the secondary microsoft webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use the current prompt and threshold
        mask, output_image = process_frame(frame, current_prompt, current_threshold, alpha_value, draw_rectangles)

        # Show the original frame and the output image with heatmap overlay
        cv2.imshow("Original", frame)
        cv2.imshow("Heatmap Overlay", output_image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create a simple GUI for prompt input and threshold slider
def gui():
    window = tk.Tk()
    window.title("Dynamic Prompt Input")

    # Set the window size to make it larger
    window.geometry("500x400")  # Width x Height

    # Prompt input field
    label_prompt = tk.Label(window, text="Enter prompt:", font=("Arial", 14))
    label_prompt.pack(pady=10)

    entry = tk.Entry(window, font=("Arial", 12))
    entry.pack(pady=10, padx=20)

    button = tk.Button(window, text="Update Prompt", command=lambda: update_prompt(entry.get()), font=("Arial", 12))
    button.pack(pady=10)

    # Threshold slider label
    label_threshold = tk.Label(window, text="Threshold:", font=("Arial", 14))
    label_threshold.pack(pady=10)

    # Threshold slider widget
    threshold_slider = tk.Scale(window, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, 
                                 command=lambda val: update_threshold(float(val)), font=("Arial", 12))
    threshold_slider.set(current_threshold)  # Set initial value to global threshold
    threshold_slider.pack(pady=10, padx=20)

    # Labels to display updated prompt and threshold
    # global prompt_label, threshold_label
    # prompt_label = tk.Label(window, text=f"Current Prompt: {current_prompt}", font=("Arial", 12))
    # prompt_label.pack(pady=10)

    # threshold_label = tk.Label(window, text=f"Current Threshold: {current_threshold:.2f}", font=("Arial", 12))
    # threshold_label.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    # Start the GUI in a separate thread
    gui_thread = threading.Thread(target=gui)
    gui_thread.start()

    # Run the main function for video processing
    main(alpha_value=0.5, draw_rectangles=True)
