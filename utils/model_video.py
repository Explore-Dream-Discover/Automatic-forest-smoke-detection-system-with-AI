import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import ssd300_vgg16,SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDHead,det_utils
from PIL import Image

# Load the model
xmodel_weights_pth = r"C:\Users\admin\Downloads\model.pth"
model=ssd300_vgg16()  # Replace YourModelClass with the class of your model
model.load_state_dict(torch.load(xmodel_weights_pth))
model.eval()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set confidence threshold
confidence_threshold = 0.5

# Open the video file
video_path = r"C:\Users\admin\Downloads\Bolivia Wildfire_ Aerials Show Wall of Fire Burning Forest.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out_path = r"C:\Users\admin\Downloads\output_video.mp4"
out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Preprocess the image
    img_tensor = to_tensor(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension if missing

    # Make predictions
    with torch.no_grad():
        # Move the input image tensor to the same device as the model
        img_tensor = img_tensor.to(device)
        
        # Get the model output
        output = model(img_tensor)
    
    # Extract the boxes, scores, and labels
    boxes = output[0]['boxes'].cpu().numpy()
    scores = output[0]['scores'].cpu().numpy()
    labels = output[0]['labels'].cpu().numpy()

    # Filter out low-confidence detections
    high_conf_indices = scores >= confidence_threshold
    boxes = boxes[high_conf_indices]
    scores = scores[high_conf_indices]
    labels = labels[high_conf_indices]

    # Overlay bounding boxes on the frame
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        # Draw bounding box on the frame
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        # Add label and score to the bounding box
        cv2.putText(frame, f'{label}: {score:.2f}', (int(xmin), int(ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
