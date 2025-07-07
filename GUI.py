import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model and class names
model = load_model("C:/Users/DESKTOP/MtechProject/mobilenetv2_bestl.keras")
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy(), predictions[0].numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255)
    return np.uint8(superimposed_img)

# GUI function
def browse_image():
    global panelA, panelB
    file_path = filedialog.askopenfilename()
    if len(file_path) > 0:
        image = Image.open(file_path).convert("RGB")
        image_resized = image.resize((256, 256))
        img_array = np.array(image_resized)

        input_img = preprocess_input(img_array.astype("float32"))
        input_img = np.expand_dims(input_img, axis=0)

        heatmap, preds = make_gradcam_heatmap(input_img, model)
        pred_label = class_names[np.argmax(preds)]

        superimposed_img = overlay_heatmap(img_array, heatmap)
        heatmap_img = Image.fromarray(superimposed_img)

        orig_img_disp = ImageTk.PhotoImage(image.resize((300, 300)))
        heatmap_disp = ImageTk.PhotoImage(heatmap_img.resize((300, 300)))

        if panelA is not None:
            panelA.destroy()
        if panelB is not None:
            panelB.destroy()

        panelA = Label(frame_left, image=orig_img_disp, bg="#f0f0f0")
        panelA.image = orig_img_disp
        panelA.pack(padx=10, pady=10)

        panelB = Label(frame_right, image=heatmap_disp, bg="#f0f0f0")
        panelB.image = heatmap_disp
        panelB.pack(padx=10, pady=10)

        result_label.config(text=f"Predicted Class: {pred_label}")

# Main window
root = tk.Tk()
root.title("Skin Lesion Classifier ")
root.geometry("700x600")
root.configure(bg="#eaf4fc")

# Top frame
top_frame = Frame(root, bg="#eaf4fc")
top_frame.pack(side="top", fill="x", pady=20)

btn_style = ttk.Style()
btn_style.configure("TButton", font=("Helvetica", 12), padding=6)
upload_btn = ttk.Button(top_frame, text="Upload Skin Image", command=browse_image)
upload_btn.pack()

# Center frames
frame_center = Frame(root, bg="#eaf4fc")
frame_center.pack(pady=10, expand=True)

frame_left = Frame(frame_center, bg="#f0f0f0", bd=2, relief="groove")
frame_left.pack(side="left", padx=20)

frame_right = Frame(frame_center, bg="#f0f0f0", bd=2, relief="groove")
frame_right.pack(side="right", padx=20)

# Bottom result label
result_label = Label(root, text="Prediction will appear here.", font=("Arial", 14), bg="#eaf4fc")
result_label.pack(pady=20)

panelA = None
panelB = None

root.mainloop()

