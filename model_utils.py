import torch
from PIL import Image
import numpy as np
import faiss
import faiss.contrib.torch_utils
import torchvision.transforms as T
from vpr_model import VPRModel

MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]

IM_SIZE = (320, 320)

def input_transform(image_size=IM_SIZE):
    return T.Compose([
		T.Resize(image_size,  interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

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

    state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['state_dict'])

    model.eval()
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model.to('cpu')

def calculate_desc(model, image, descriptor_path):
    rgb_image = image
    desc_img = Image.fromarray(np.uint8(rgb_image))
    with torch.no_grad():
        desc = model(input_transform()(desc_img).unsqueeze(0))
    descriptors = torch.load(descriptor_path)
    candidate = search_candidate(desc, descriptors)
    return candidate

def search_candidate(query_descriptor, ref_descriptors):
    r_list = ref_descriptors
    q = query_descriptor
    embed_size = r_list.shape[1]
    r_list = r_list.to('cpu')
    q = q.to('cpu')
    faiss_index = faiss.IndexFlatL2(embed_size)
    # add references
    faiss_index.add(r_list)

    # search for queries in the index
    similarities, prediction = faiss_index.search(q, 1)
    print("PRED:", prediction, similarities)
    return prediction