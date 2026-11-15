# MIT License for the portions adapted from the credited upstream projects.
#
# Copyright (c) 2023 anosorae
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import os.path as osp

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from SCOPE.utils.iotools import read_image, read_json
from SCOPE.utils.simple_tokenizer import SimpleTokenizer


TRI_DATASETS = {
    "Tri-ICFG-PEDES": {
        "directory": "ICFG-PEDES",
        "annotation": "ICFG-PEDES.json",
        "image_directory": "imgs",
        "sketch_directory": "imgs-sketch",
        "path_key": "file_path",
    },
    "Tri-CUHK-PEDES": {
        "directory": "CUHK-PEDES",
        "annotation": "reid_raw.json",
        "image_directory": "imgs",
        "sketch_directory": "imgs-sketch2",
        "path_key": "file_path",
    },
    "Tri-RSTPReid": {
        "directory": "RSTPReid",
        "annotation": "data_captions.json",
        "image_directory": "imgs",
        "sketch_directory": "img-sketch2",
        "path_key": "img_path",
    },
}

TRI_ALIASES = {
    "ICFG-PEDES": "Tri-ICFG-PEDES",
    "CUHK-PEDES": "Tri-CUHK-PEDES",
    "RSTPReid": "Tri-RSTPReid",
}


def canonical_tri_name(dataset_name):
    return TRI_ALIASES.get(dataset_name, dataset_name)


def is_tri_dataset(dataset_name):
    return canonical_tri_name(dataset_name) in TRI_DATASETS


def tokenize(caption, tokenizer, text_length=77):
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot] + tokenizer.encode(caption) + [eot]
    if len(tokens) > text_length:
        tokens = tokens[:text_length]
        tokens[-1] = eot
    result = torch.zeros(text_length, dtype=torch.long)
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class TriPersonDataset:
    def __init__(self, root, dataset_name):
        self.name = canonical_tri_name(dataset_name)
        if self.name not in TRI_DATASETS:
            raise ValueError(f"Unsupported Tri dataset: {dataset_name}")

        spec = TRI_DATASETS[self.name]
        dataset_root = osp.join(root, spec["directory"])
        annotation_path = osp.join(dataset_root, spec["annotation"])
        if self.name == "Tri-ICFG-PEDES" and not osp.isfile(annotation_path):
            alternate = osp.join(dataset_root, "ICFG_PEDES.json")
            if osp.isfile(alternate):
                annotation_path = alternate
        if not osp.isfile(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        self.image_root = osp.join(dataset_root, spec["image_directory"])
        self.sketch_root = osp.join(dataset_root, spec["sketch_directory"])
        if not osp.isdir(self.image_root) or not osp.isdir(self.sketch_root):
            raise FileNotFoundError(
                f"Expected image and sketch directories under {dataset_root}"
            )

        annotations = read_json(annotation_path)
        split_annotations = {
            split: [item for item in annotations if item["split"] == split]
            for split in ("train", "val", "test")
        }

        self.train, self.num_train_ids = self._build_train(
            split_annotations["train"], spec["path_key"]
        )
        self.val = self._build_eval(split_annotations["val"], spec["path_key"])
        self.test = self._build_eval(split_annotations["test"], spec["path_key"])

    def _paths(self, item, path_key):
        relative_path = item[path_key]
        return (
            osp.join(self.image_root, relative_path),
            osp.join(self.sketch_root, relative_path),
        )

    def _build_train(self, annotations, path_key):
        raw_ids = sorted({int(item["id"]) for item in annotations})
        pid_map = {pid: index for index, pid in enumerate(raw_ids)}
        samples = []
        for image_id, item in enumerate(annotations):
            rgb_path, sketch_path = self._paths(item, path_key)
            pid = pid_map[int(item["id"])]
            for caption in item["captions"]:
                samples.append((pid, image_id, rgb_path, sketch_path, caption))
        return samples, len(raw_ids)

    def _build_eval(self, annotations, path_key):
        data = {
            "image_pids": [], "image_paths": [], "image_ids": [],
            "sketch_pids": [], "sketch_paths": [], "sketch_ids": [],
            "caption_pids": [], "captions": [], "caption_sketch_paths": [],
            "caption_image_ids": [],
        }
        for image_id, item in enumerate(annotations):
            pid = int(item["id"])
            rgb_path, sketch_path = self._paths(item, path_key)
            data["image_pids"].append(pid)
            data["image_paths"].append(rgb_path)
            data["image_ids"].append(image_id)
            data["sketch_pids"].append(pid)
            data["sketch_paths"].append(sketch_path)
            data["sketch_ids"].append(image_id)
            for caption in item["captions"]:
                data["caption_pids"].append(pid)
                data["captions"].append(caption)
                data["caption_sketch_paths"].append(sketch_path)
                data["caption_image_ids"].append(image_id)
        return data


class TriTrainDataset(Dataset):
    def __init__(self, samples, transform, text_length):
        self.samples = samples
        self.transform = transform
        self.text_length = text_length
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pid, image_id, rgb_path, sketch_path, caption = self.samples[index]
        return {
            "pids": pid,
            "image_ids": image_id,
            "rgbs": self.transform(read_image(rgb_path)),
            "sks": self.transform(read_image(sketch_path)),
            "caption_ids": tokenize(caption, self.tokenizer, self.text_length),
        }


class TriGalleryDataset(Dataset):
    def __init__(self, pids, paths, image_ids, transform):
        self.pids, self.paths, self.image_ids = pids, paths, image_ids
        self.transform = transform

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        return self.pids[index], self.transform(read_image(self.paths[index])), self.image_ids[index]


class TriSketchDataset(TriGalleryDataset):
    pass


class TriTextSketchDataset(Dataset):
    def __init__(self, data, transform, text_length):
        self.data = data
        self.transform = transform
        self.text_length = text_length
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.data["caption_pids"])

    def __getitem__(self, index):
        sketch = self.transform(read_image(self.data["caption_sketch_paths"][index]))
        caption = tokenize(self.data["captions"][index], self.tokenizer, self.text_length)
        return (
            sketch,
            self.data["caption_image_ids"][index],
            self.data["caption_pids"][index],
            caption,
        )


def build_transforms(img_size, augment=False, training=True):
    height, width = img_size
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    transforms = [T.Resize((height, width))]
    if training:
        transforms.append(T.RandomHorizontalFlip(0.5))
    if training and augment:
        transforms.extend([T.Pad(10), T.RandomCrop((height, width))])
    transforms.extend([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    if training and augment:
        transforms.append(T.RandomErasing(scale=(0.02, 0.4), value=mean))
    return T.Compose(transforms)


def collate(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        if torch.is_tensor(batch[0][key])
        else torch.tensor([item[key] for item in batch])
        for key in batch[0]
    }


