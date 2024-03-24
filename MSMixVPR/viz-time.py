import glob
import os
import time
from typing import Tuple

import torch
from PIL import Image
from torch.utils import data
import numpy as np
import torchvision.transforms as tvf
from tqdm import tqdm
import cv2
import faiss
from sys import getsizeof

from main import VPRModel
from dataloaders import PittsburgDataset, MapillaryDataset, NordlandDataset, SPEDDataset

class InferencePipeline:
    def __init__(self, model, dataset, feature_dim, batch_size=4, num_workers=4, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.extraction_time = 0
        self.dataloader = data.DataLoader(self.dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=self.num_workers,
                                          pin_memory=True,
                                          drop_last=False)

    def run(self, split: str = 'all') -> np.ndarray:

        # if os.path.exists(f'./DemoLogs/global_descriptors_{split}.npy'):
        #     print(f"Skipping {split} features extraction, loading from cache")
        #     return np.load(f'./DemoLogs/global_descriptors_{split}.npy')

        self.model.to(self.device)
        with torch.no_grad():
            global_descriptors = np.zeros((len(self.dataset), self.feature_dim))
            for batch in tqdm(self.dataloader, ncols=100, desc=f'Extracting {split} features'):
                imgs, indices = batch
                imgs = imgs.to(self.device)

                # model inference
                start = time.time()
                descriptors = self.model(imgs)
                end = time.time()
                self.extraction_time += (end - start)
                descriptors = descriptors.detach().cpu().numpy()

                # add to global descriptors
                global_descriptors[np.array(indices), :] = descriptors

        # save global descriptors
        np.save(f'./DemoLogs/global_descriptors_{split}.npy', global_descriptors)
        
        self.extraction_time = self.extraction_time / len(self.dataset)
        return global_descriptors


def load_image(path):
    image_pil = Image.open(path).convert("RGB")

    # add transforms
    transforms = tvf.Compose([
        tvf.Resize((320, 320), interpolation=tvf.InterpolationMode.BICUBIC),
        tvf.ToTensor(),
        tvf.Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
    ])

    # apply transforms
    image_tensor = transforms(image_pil)
    return image_tensor


def load_model_musc(ckpt_path):
    # Note that images must be resized to 320x320
    model = VPRModel(backbone_arch='resnet50musc',
                     layers_to_crop=[4],
                     agg_arch='MuscVPR',
                     agg_config={
                            'in_h' : 40,
                            'in_w' : 40,
                            'in_h_1' : 20,
                            'in_w_1' : 20,
                            'in_channels' : 2048,
                            'out_channels' : 1024,
                            'mix_depth' : 2,
                            'mlp_ratio' : 1,
                            'out_rows' : 4}, # the output dim will be (out_rows * out_channels)
                    )

    state_dict = torch.load(ckpt_path)
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

def calculate_top_k(descriptors: np.ndarray,
                    dataset_name,
                    dataset,
                    top_k: int = 1) -> np.ndarray:
    if dataset_name == 'nordland' or dataset_name == 'sped':
        num_ref = dataset.num_references
        gt = dataset.ground_truth
    elif dataset_name == 'msls':
        num_ref = dataset.num_references
        gt = dataset.pIdx
    else:
        num_ref = dataset.dbStruct.numDb
        gt = dataset.getPositives()

    r_list = np.float32(descriptors[ : num_ref])
    q_list = np.float32(descriptors[num_ref : ])
    embed_size = r_list.shape[1]
    faiss_index = faiss.IndexFlatL2(embed_size)
        
    # add references
    faiss_index.add(r_list)

    # search for queries in the index
    start = time.time()
    _, predictions = faiss_index.search(q_list, top_k)
    end = time.time()

    matching_time = (end - start) / len(q_list)

    return predictions, matching_time

def record_matches(top_k_matches: np.ndarray,
                   dataset,
                   dataset_name,
                   out_file: str = 'record.txt') -> None:
    # print("TOPK", top_k_matches)
    if dataset_name == 'pittsburgh':
        ground_truth = dataset.getPositives()
        num_ref = dataset.dbStruct.numDb
    elif dataset_name == 'msls':
        ground_truth = dataset.pIdx
        num_ref = dataset.num_references
    else:
        ground_truth = dataset.ground_truth
        num_ref = dataset.num_references
    with open(f'{out_file}', 'a') as f:
        k_values=[1, 5, 10]
        correct_at_k = 0
        for query_index, pred in enumerate(top_k_matches):
            pred_query_path = dataset.images[num_ref + query_index]
            status = False
            if np.any(np.in1d(pred[:len(pred)], ground_truth[query_index])):
                status = True
                correct_at_k += 1
            pred_db_paths = dataset.images[pred[0]]
            # f.write(f'{pred_query_path} {pred_db_paths} {status}\n')
            # break
        correct_at_k = (correct_at_k / len(top_k_matches)) * 100
        print("Recall@1:", correct_at_k)

def visualize(top_k_matches: np.ndarray,
              dataset,
              dataset_name,
              visual_dir: str = './DemoLogs/visualize',
              img_resize_size: Tuple = (320, 320)) -> None:
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    if dataset_name == 'pittsburgh':
        DATASET_ROOT = ''
        num_ref = dataset.dbStruct.numDb
        ground_truth = dataset.getPositives()
    elif dataset_name == 'msls':
        DATASET_ROOT = '/root/MultiScale-MixVPR/datasets/msls-val/'
        num_ref = dataset.num_references
        ground_truth = dataset.pIdx
    else:
        if dataset_name == 'nordland':
            DATASET_ROOT = '/root/MultiScale-MixVPR/datasets/Nordland/'
        else:
            DATASET_ROOT = '/root/MultiScale-MixVPR/datasets/SPEDTEST/'
        num_ref = dataset.num_references
        ground_truth = dataset.ground_truth
    for query_index, pred in enumerate(top_k_matches):
        pred_q_path = f'{DATASET_ROOT}{dataset.images[num_ref + query_index]}'
        q_array = cv2.imread(pred_q_path, cv2.IMREAD_COLOR)
        q_array = cv2.resize(q_array, img_resize_size, interpolation=cv2.INTER_CUBIC)
        gap_array = np.ones((340, 10, 3)) * 255  # white gap

        pred_db_paths = f'{DATASET_ROOT}{dataset.images[pred[0]]}'
        db_array = cv2.imread(pred_db_paths, cv2.IMREAD_COLOR)
        db_array = cv2.resize(db_array, img_resize_size, interpolation=cv2.INTER_CUBIC)
        # Define the border size and color
        border_size = 10
        if np.any(np.in1d(pred[:len(pred)], ground_truth[query_index])):
            border_color = [0, 255, 0]  # Green
        else:
            border_color = [0, 0, 255]  # Red
        db_array = cv2.copyMakeBorder(
            db_array,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color
        )
        q_array = cv2.copyMakeBorder(
            q_array,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        q_array = np.concatenate((q_array, gap_array, db_array), axis=1)

        result_array = q_array.astype(np.uint8)

        # Save the result array as an image using cv2
        cv2.imwrite(f'{visual_dir}/{os.path.basename(pred_q_path)}', result_array)

def main():
    dataset_name = 'nordland'

    if dataset_name == 'nordland':
        dataset = NordlandDataset.NordlandDataset(input_transform=tvf.Compose([
        tvf.ToTensor(),
        tvf.Resize((320, 320)),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
)
    elif dataset_name == 'sped':
        dataset = SPEDDataset.SPEDDataset(input_transform=tvf.Compose([
        tvf.ToTensor(),
        tvf.Resize((320, 320)),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
)
    elif dataset_name == 'pittsburgh':
        dataset = PittsburgDataset.get_whole_test_set(input_transform=tvf.Compose([
        tvf.ToTensor(),
        tvf.Resize((320, 320)),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
)
    elif dataset_name == 'msls':
        dataset = MapillaryDataset.MSLS(input_transform=tvf.Compose([
        tvf.ToTensor(),
        tvf.Resize((320, 320)),
        tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
)

    # load model
    # model = load_model_mix('/root/MultiScale-MixVPR/LOGS/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt')
    # model = load_model_convap('/root/MultiScale-MixVPR/LOGS/resnet50_ConvAP_2048_2x2.ckpt')
    # model = load_model_cosplace()
    model = load_model_musc('/root/MultiScale-MixVPR/LOGS/saved_ckpt/msmix_weights.ckpt')

    # set up inference pipeline
    inference_pipeline = InferencePipeline(model=model, dataset=dataset, feature_dim=4096)

    global_descriptors = inference_pipeline.run()

    # calculate top-k matches
    top_k_matches, matching_time = calculate_top_k(global_descriptors, dataset_name, dataset, top_k=1)

    # record query_database_matches
    record_matches(top_k_matches, dataset, dataset_name, out_file='./DemoTimeLogs/record_mix_sped_test.txt')

    # # visualize top-k matches
    # visualize(top_k_matches, dataset, dataset_name, visual_dir='./DemoTimeLogs/visualize/mix_sped')

    print("Image encoding time (s):", inference_pipeline.extraction_time)
    print("Image matching time (s):", matching_time)
    print("Image retrieval time (s):", inference_pipeline.extraction_time + matching_time)


if __name__ == '__main__':
    main()
