import sys
sys.path.append('..') # append parent directory, we need it
# sys.path.insert(0, '/home/vpr-training/test_ai/MuscVPR')

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from utils.validation import get_validation_recalls

import time
import psutil

torch.backends.cudnn.benchmark = True

MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]

IM_SIZE = (320, 320)

def input_transform(image_size=IM_SIZE):
    return T.Compose([
		T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

from dataloaders.NordlandDataset import NordlandDataset
from dataloaders.SPEDDataset import SPEDDataset
from dataloaders import PittsburgDataset
from dataloaders.MapillaryDataset import MSLS

def get_val_dataset(dataset_name, input_transform=input_transform()):
    dataset_name = dataset_name.lower()
    
    if 'nordland' in dataset_name:    
        ds = NordlandDataset(input_transform = input_transform)
    
    elif 'sped' in dataset_name:
        ds = SPEDDataset(input_transform = input_transform)
    
    elif 'msls' in dataset_name:
        ds = MSLS(input_transform = input_transform)

    elif 'pitts30k' in dataset_name:
        ds = PittsburgDataset.get_whole_test_set(input_transform=input_transform)
        
    else:
        raise ValueError
    
    if 'pitts' in dataset_name:
        num_references = ds.dbStruct.numDb
        num_queries = len(ds)-num_references
        ground_truth = ds.getPositives()
    elif 'msls' in dataset_name:
        num_references = ds.num_references
        num_queries = len(ds)-num_references
        ground_truth = ds.pIdx
    else:
        num_references = ds.num_references
        num_queries = ds.num_queries
        ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth
    
def get_descriptors(model, dataloader, device):
    descriptors = []
    extraction_latency = []
    # cpu_usage = []
    with torch.no_grad():
        for batch in tqdm(dataloader, 'Calculating descriptors...'):
            places, label = batch
            imgs, labels = batch
            imgs = imgs.to(device)
            start = time.time()
            output = model(imgs)
            end = time.time()
            descriptors.append(output)

    return torch.cat(descriptors)

from main import VPRModel

def load_model_musc(ckpt_path):
    # Note that images must be resized to 320x320
    model = VPRModel(backbone_arch='resnet50musc',
                     layers_to_crop=[4],
                     agg_arch='MuscVPR',
                     agg_config={'in_channels' : 2048,
                            'in_h' : 40,
                            'in_w' : 40,
                            # 'in_channels_1' : 2048,
                            'in_h_1' : 20,
                            'in_w_1' : 20,
                            'out_channels' : 1024,
                            # 'out_channels_1' : 1024,
                            'mix_depth' : 2,
                            'mlp_ratio' : 1,
                            'out_rows' : 4}, # the output dim will be (out_rows * out_channels)
                     )

    state_dict = torch.load(ckpt_path, map_location=torch.device('cuda'))
    model.load_state_dict(state_dict['state_dict'])

    model.eval()
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model

def load_model_mix(ckpt_path):
    # Note that images must be resized to 320x320
    model = VPRModel(
                backbone_arch='resnet50',
                layers_to_crop=[4],
                agg_arch='MixVPR',
                agg_config={'in_channels' : 1024,
                'in_h' : 20,
                'in_w' : 20,
                'out_channels' : 1024,
                'mix_depth' : 4,
                'mlp_ratio' : 1,
                'out_rows' : 4}, # the output dim will be (out_rows * out_channels)
    )

    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_model_convap(ckpt_path):
    # Note that images must be resized to 320x320
    model = VPRModel(
                    backbone_arch='resnet50',
                    layers_to_crop=[],
                    agg_arch='ConvAP',
                    agg_config={
                        'in_channels': 2048,
                        'out_channels': 2048},
    )

    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_model_cosplace():
    model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
    model.eval()
    return model

device = torch.device("cuda:0")
# device = torch.device("cpu")

# model = load_model_cosplace()
model = load_model_musc('/root/MultiScale-MixVPR/LOGS/saved_ckpt/msmix_weights.ckpt')
model = model.to(device)

# val_dataset_names = ['MSLS-val', 'SPED', 'Nordland', 'pitts30k_test']
val_dataset_names = ['Nordland']
batch_size = 64

for val_name in val_dataset_names:
    val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name)
    val_loader = DataLoader(val_dataset, num_workers=16, batch_size=batch_size)
    print(f'Evaluating on {val_name}')
    descriptors = get_descriptors(model, val_loader, device)
    r_list = descriptors[ : num_references]
    q_list = descriptors[num_references : ]
    recalls_dict = get_validation_recalls(r_list=r_list,
                                                q_list=q_list,
                                                k_values=[1, 5, 10, 15, 20, 25, 50, 75, 100],
                                                gt=ground_truth,
                                                print_results=True,
                                                dataset_name=val_name,
                                                faiss_gpu=False
                                                )
    del descriptors
    print('========> DONE!\n\n')
