from fastapi import FastAPI
import base64
import pickle
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
import numpy as np
from torch import nn
import torch

# needed to be able to import the model. Getting attribute reference error without this class here
class CNNnet(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 32, 2, stride=1, padding=0)
        self.fc1 = nn.Linear(32*12*12, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
        return x

models_dir = './tmp/fhe_models/'
fhe_model = FHEModelServer(path_dir=models_dir)
fhe_model.load()
model = CNNnet(8)
model.load_state_dict(torch.load('./tmp/models/cnn.pt'))

app = FastAPI()
classes =['PA-cutaneous-larva-migrans', 'VI-shingles', 'FU-ringworm', 'FU-athlete-foot', 'BA- cellulitis', 'FU-nail-fungus', 'BA-impetigo', 'VI-chickenpox']

def decode(el):
    return base64.b64decode(el.encode("utf-8"))

@app.post("/predict/fhe")
async def predict_fhe(data: dict):
    request = data["data"]
    encrypted_img = decode(request["image"])
    eval_key = decode(request["key"])
    encrypted_res = fhe_model.run(encrypted_img, eval_key)

    return {"prediction": encrypted_res}

@app.post("/predict")
async def predict(data: dict):
    image = torch.tensor(np.array(data["data"]))
    prediction = model.forward(image.float())
    result = classes[prediction.argmax(1).detach().tolist()[0]]

    return {"prediction": result}
