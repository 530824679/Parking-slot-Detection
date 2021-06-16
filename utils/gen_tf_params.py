
import torch
import pickle

pt_file = r'/home/chenwei/HDD/Project/2D_ObjectDetect/pt_weight/archive/data.pkl'

device = torch.device("cpu")
model = torch.load(pt_file, map_location=device)
model = model['model'].float()
model.to(device)
model.eval()

data_dict = {}
for k, v in model.state_dict().item():
    vr = v.cpu().numpy()
    data_dict[k] = vr

print(data_dict['model.24.anchors'])
print(data_dict['model.24.anchor_grid'])
fid=open('params.dict','wb')
pickle.dump(data_dict,fid)
fid.close()