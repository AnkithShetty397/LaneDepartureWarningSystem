import cv2
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms as transforms
import numpy as np
import logging

#loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet("resnet18", in_channels=3, classes=1, activation=None).to(device)
checkpoint_path = "/model/unet_resnet18.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#capture settings
cap = cv2.VideoCapture(0)

while not cap.isOpened():
    pass

cap.set(cv2.CAP_PROP_FPS, 5)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

#driving region mask
x0, x1, x2, x3 = 380,960-380,960-300,300
y0, y1 = 520, 640

y, x = np.ogrid[:640, :960]

condition = (x >= x0 + (x3 - x0) / (y1 - y0) * (y - y0)) & (x < x1 + (x2 - x1) / (y1 - y0) * (y - y0)) & (y >= y0) & (y < y1)

#transform the frame
transform = transforms.Compose([
        transforms.ToTensor()
    ])

#configuring logging module
log_filename = "/edge_device/log/error_log.txt"
logging.basicConfig(filename=log_filename, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def warning_func():
    #audio or visual warning 
    pass        

def system_malfunction_warning():
    #warning light on dashboard
    pass

try:
    while True: 
        ret,frame=cap.read()

        if not ret:
            system_malfunction_warning()
            continue

        input_tensor = transform(frame).to(device)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)

        output_image = output.squeeze().detach().cpu().numpy()
        output_image = ((output_image - output_image.min()) / (output_image.max() - output_image.min()))
        output_image[output_image>0.80]=255

        mask = np.zeros((640, 960), dtype=np.uint8)
        mask = torch.zeros((1, 640, 960), dtype=torch.uint8).to(device)

        mask[0, condition] = 1
        res = output_image * mask.cpu().numpy()
        sum_value = np.sum(res)

        if sum_value>1:
            warning_func()
        
        if cv2.waitKey(1) & 0xFF == ord('Q'):
            break
        
except Exception as e:
    logging.error(f"An unexpected exception occurred: {e}")

finally: 
    cap.release()

    
