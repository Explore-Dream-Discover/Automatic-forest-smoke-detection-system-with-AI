import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import ssd300_vgg16,SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDHead,det_utils
from PIL import Image
from io import BytesIO



model_weights_pth = r"C:\Users\admin\Downloads\model.pth"
#label 0 is fixed for background
classes=["background","smoke"]

num_classes=2
device="cuda" if torch.cuda.is_available() else "cpu"
model=ssd300_vgg16()

in_channels=det_utils.retrieve_out_channels(model.backbone,(480,480))
num_anchors=model.anchor_generator.num_anchors_per_location()
model.head=SSDHead(in_channels=in_channels,num_anchors=num_anchors,
                   num_classes=num_classes)

model.load_state_dict(torch.load(model_weights_pth,map_location=device))
model.to(device)


st.title("Automatic-forest-smoke-detection-system-with-AI")


# Image uploader
uploaded_file = st.file_uploader("Choose an image file to predict", type=["jpg", "jpeg", "png","PNG"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Open the uploaded image file
   
    image = Image.open(uploaded_file).convert("RGB")
    # Preprocess the image
    img_tensor = to_tensor(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension if missing

    # Make predictions
    model.eval()
    with torch.no_grad():
        # Move the input image tensor to the same device as the model
        img_tensor = img_tensor.to(device)
        
        # Get the model output
        output = model(img_tensor)
        
    # Move the predicted bounding box coordinates to CPU
    prediction = output[0]  # Since we have only one image, take the first prediction

    # Extract the boxes, scores, and labels
    boxes = output[0]['boxes'].cpu().numpy()
    scores = output[0]['scores'].cpu().numpy()
    labels = output[0]['labels'].cpu().numpy()

    # Set confidence threshold
    confidence_threshold = 0.5

    # Filter out low-confidence detections
    high_conf_indices = scores >= confidence_threshold
    boxes = boxes[high_conf_indices]
    scores = scores[high_conf_indices]
    labels = labels[high_conf_indices]

    # Plot the image and overlay the bounding boxes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # Add bounding boxes to the plot
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        
        # Create a rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
    # Add label and score to the bounding box
    # Add label and score to the bounding box
        if score <= 1:
            ax.text(xmin, ymin, f'{label}: Forest Smoke', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        else:
            ax.text(xmin, ymin, f'{label}:Background', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

        # Save the plot to a BytesIO object
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        # Display the plot in Streamlit
        st.image(buf, caption="Image", use_column_width=True)





else:
    st.write("Please upload an image file to proceed.")
