from fastapi import FastAPI, UploadFile, File, APIRouter
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms 

from PIL import Image 
import io 
import os

app = FastAPI(title="Number Guesser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


'''
Define Transform to manipulate data
'''
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


'''
Define Model Architecture
'''
class MNISTnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(784, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.log_softmax(self.output(x), dim=1)


device = torch.device("cpu")
model = MNISTnet()

model.load_state_dict(torch.load("model/mnist_model.pth", map_location=device, weights_only=True))
model.eval()

'''
Deploy settings
'''
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dir = os.path.join(base_dir, "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


'''
API
'''
@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read() 
        image = Image.open(io.BytesIO(image_bytes))

        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            prediction = torch.argmax(output, dim=1).item()
            
        return {"filename": file.filename, "predicted_number": prediction}
    
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Error processing image: {str(e)}"})