import cv2
from torch import hub
import torch
from torch.nn import functional as F
from torch.autograd import Variable as V
import numpy
from time import time
from torchvision import transforms

def score_frame(frame, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    return labels, cord

def plot_boxes(results, frame, model):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, model.names[int(i)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame

def load_model():
    down = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return down


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tensor = transforms.ToTensor()

player = cv2.VideoCapture(0)
x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
model = load_model()

while True:
    start_time = time()
    ret, frame = player.read()
    assert ret
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tensor(frame)
    frame = img.permute(1,2,0).contiguous().numpy()

    pred = model(V(img.unsqueeze(0)))[-1][0].cpu()
    # frame = [frame]
    # results = model(frame)
    # labels = results.xyxyn[0][:, -1].numpy()
    # cord = results.xyxyn[0][:, :-1].numpy()
    # result = labels, cord

    results = score_frame(frame, model)
    frame = plot_boxes(results, frame, model)

    pred = pred.data.max(0)[0].numpy()
    #pred = cv2.resize(pred, dsize=(args.size, args.size))
    
    pred = cv2.resize(pred, dsize=(x_shape, y_shape), interpolation=cv2.INTER_CUBIC)

    # frame[:,:,1] += pred
    
    cv2.imshow('frame', frame)
    cv2.imshow('pred', pred)
    cv2.waitKey(1)

    # results = score_frame(frame, model)
    # frame = plot_boxes(results, frame, model)
    # end_time = time()

    # cv2.imshow("Output", player)