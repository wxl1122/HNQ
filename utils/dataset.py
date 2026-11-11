r""" Few-Shot Anomaly Detection Dataset """

import os
import json
import random
import hashlib
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import PIL.Image as Image
import numpy as np

class FSDataset(Dataset):
    def __init__(self,
                 data_root:str ='/data/datasets',
                 data_mode:str = 'mvtec_visa',
                 data_name_json:str ='meta.json',
                 fold:int =0,
                 split:str ='train',
                 shot=(5, 1),
                 transform:object =None,
                 choice=2000,
                 seed=111,
                 eval_label_sampling='balanced',
                 normal_support_split='test',
                 image_cache_size=0,
                 mask_size=None):
        self.mask_size = (mask_size, mask_size) if mask_size else None
        if image_cache_size < 0:
            raise ValueError('image_cache_size must be nonnegative')
        self.image_cache_size = image_cache_size
        self._image_cache = OrderedDict()
        if split not in ['train', 'eval', 'test', 'source_eval']:
            raise ValueError(f'Unknown split: {split}')
        self.split = 'train' if split == 'train' else 'val'
        self.source_eval = split == 'source_eval'
        self.data_root = data_root
        data = ['mvtec', 'visa'] if data_mode == 'mvtec_visa' else ['Real-IAD/realiad_1024_unzip']
        # data = ['mvtec', 'visa'] if data_mode == 'mvtec_visa' else ['realiad']
        self.data_mode = data_mode
        self.data_name_json = data_name_json

        self.choice = choice
        self.seed = seed
        if eval_label_sampling not in ('balanced', 'natural'):
            raise ValueError(f'Unknown evaluation label sampling: {eval_label_sampling}')
        self.eval_label_sampling = eval_label_sampling
        if normal_support_split not in ('train', 'test'):
            raise ValueError(f'Unknown normal support split: {normal_support_split}')
        self.normal_support_split = normal_support_split
        # self.nfolds = 2
        self.fold = fold
        self.n_shot, self.a_shot = shot

        self.transform = transform

        self.initialize(data)

        self.class_ids = self.build_class_ids()

    def initialize(self, data):
        # Generate metadata
        self.metadata = {}
        self.train_normal_supports = {}
        self.all_product = []
        for d in data:
            data_path = os.path.join(self.data_root, d)
            with open(os.path.join(data_path, self.data_name_json)) as f:
                # data_info.update(json.load(f)['test'])
                data_info = json.load(f)

            if self.normal_support_split == 'train':
                for product, samples in data_info['train'].items():
                    self.train_normal_supports[product] = [
                        {'img_path': os.path.join(data_path, sample['img_path']),
                         'mask_path': os.path.join(data_path, sample['mask_path']),
                         'anomaly': sample['anomaly']}
                        for sample in samples if not sample['anomaly']]
            data_info = data_info['test']
            products = list(data_info.keys())
            products.sort()
            self.all_product += products

            for product in products:
                self.metadata.setdefault(product, {'normal': {}, 'abnormal': {}})

                for sample in data_info[product]:
                    sample_info = {
                        'img_path': os.path.join(data_path, sample['img_path']),
                        'mask_path': os.path.join(data_path, sample['mask_path']),
                        'anomaly': sample['anomaly']
                    }

                    if sample['anomaly']:
                        product_type = 'abnormal'
                        if sample['specie_name'] == '':
                            sample['specie_name'] = 'bad'
                    else:
                        product_type = 'normal'
                        if sample['specie_name'] == '':
                            sample['specie_name'] = 'good'
                    self.metadata[product][product_type].setdefault(sample['specie_name'], []).append(sample_info)

        self.nclass = len(self.all_product)

    def __len__(self):
        # return self.choice
        return self.choice

    def mask_list_transform(self, mask_list, img_size):
        mask_tensor = []
        for mask in mask_list:
            mask_tensor.append(F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), img_size, mode='nearest').squeeze())
        return torch.stack(mask_tensor)

    def image_list_transform(self, image_list):
        image_tensor = []
        for image in image_list:
            image_tensor.append(image if torch.is_tensor(image) else self.transform(image))
        return torch.stack(image_tensor)

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # tuple: (image_list, mask_list, anomaly_list)
        query_tuple, support_normal_tuple, support_abnormal_tuple, sample_product = self.load_frame(idx)

        query_image = self.image_list_transform(query_tuple[0]) # [1, C, H, W]
        query_mask = self.mask_list_transform(query_tuple[1], getattr(self, 'mask_size', None) or query_image.shape[-2:]) # [1, H, W]
        query_data = [query_image, query_mask]

        support_normal_data = [0, 0]
        support_abnormal_data = [0, 0]

        if self.n_shot:
            support_normal_imgs = self.image_list_transform(support_normal_tuple[0])
            support_normal_masks = self.mask_list_transform(support_normal_tuple[1], support_normal_imgs.size()[-2:])
            support_normal_data = [support_normal_imgs, support_normal_masks]

        if self.a_shot:
            support_abnormal_imgs = self.image_list_transform(support_abnormal_tuple[0])
            support_abnormal_masks = self.mask_list_transform(support_abnormal_tuple[1], support_abnormal_imgs.size()[-2:])
            support_abnormal_data = [support_abnormal_imgs, support_abnormal_masks]

        ret_dict = {
            'query': query_data,
            'image_level_label': torch.tensor(query_tuple[2][0], dtype=torch.long),
            'support_normal': support_normal_data,
            'support_abnormal': support_abnormal_data,
            'sample_product': sample_product
        }

        return ret_dict

    def build_class_ids(self):
        if self.data_mode == 'mvtec_visa': # mvtec and visa
            class_ids_val = range(0, 15) if self.fold == 0 else range(15, 27)
            class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        else: # RealIAD
            self.nfolds = 2
            # number of product categories
            nclass_trn = self.nclass // self.nfolds
            class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
            class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        class_trn = [self.all_product[i] for i in class_ids_trn]
        class_val = [self.all_product[i] for i in class_ids_val]
        class_ids = class_ids_trn if self.split == 'train' or self.source_eval else class_ids_val

        if self.source_eval:
            msg = f'Source-eval classes: {class_trn}'
        else:
            msg = f'Train classes: {class_trn}' if self.split == 'train' else f'Val classes: {class_val}'
        print(msg)

        self.products = class_trn if self.split == 'train' or self.source_eval else class_val

        return class_ids

    def random_sample(self, sample_list, num, rng=random):
        if num == 0:
            return [], []
        if len(sample_list) < num:
            raise ValueError(f'Requested {num} samples from a pool of {len(sample_list)}')
        selected_idx = rng.sample(range(len(sample_list)), num)
        selected_sample = [sample_list[i] for i in selected_idx]
        return selected_idx, selected_sample

    def get_sample_info(self, sample_dict):
        img_path = sample_dict['img_path']
        mask_path = sample_dict['mask_path']
        anomaly = sample_dict['anomaly']
        return img_path, mask_path, anomaly

    def read_mask(self, anomaly, mask_path, imsize):
        if not anomaly:
            # PIL reports image size as (width, height), tensors use (height, width).
            mask = torch.zeros((imsize[1], imsize[0]))
        else:
            mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
            mask[mask>0.5] = 1
        return mask

    def read_data(self, data_list):
        image_list = []
        mask_list = []
        anomaly_list = []
        for data in data_list:
            data_path, mask_path, anomaly = self.get_sample_info(data)
            if getattr(self, 'image_cache_size', 0):
                key = (data_path, mask_path, anomaly)
                if key not in self._image_cache:
                    image = Image.open(data_path).convert('RGB')
                    mask = self.read_mask(anomaly, mask_path, image.size)
                    tensor = self.transform(image)
                    resized_mask = self.mask_list_transform(
                        [mask], getattr(self, 'mask_size', None) or tensor.shape[-2:])[0]
                    self._image_cache[key] = (tensor, resized_mask)
                cached_image, cached_mask = self._image_cache[key]
                self._image_cache.move_to_end(key)
                while len(self._image_cache) > self.image_cache_size:
                    self._image_cache.popitem(last=False)
                image_list.append(cached_image)
                mask_list.append(cached_mask)
            else:
                image_list.append(Image.open(data_path).convert('RGB'))
                mask_list.append(self.read_mask(anomaly, mask_path, image_list[-1].size))
            anomaly_list.append(anomaly)
        return image_list, mask_list, anomaly_list

    def load_frame(self, idx):
        # Validation episodes must not change between epochs; otherwise model
        # selection compares checkpoints on different samples. Training remains
        # stochastic (DataLoader seeds each worker independently).
        rng = random if self.split == 'train' else random.Random(self.seed + idx)

        if self.split == 'train':
            sample_product_idx = rng.choice(list(self.class_ids))
            query_type = rng.choice(['normal', 'abnormal'])
        else:
            sample_product_idx = self.class_ids[idx % len(self.class_ids)]
            query_type = ['normal', 'abnormal'][(idx // len(self.class_ids)) % 2]
        sample_product = self.all_product[sample_product_idx]

        normal_species = sorted(self.metadata[sample_product]['normal'])
        abnormal_species = sorted(self.metadata[sample_product]['abnormal'])
        support_normal_specie_type = rng.choice(normal_species)
        support_abnormal_specie_type = rng.choice(abnormal_species)

        if self.split == 'val' and self.eval_label_sampling == 'natural':
            # Sample uniformly over actual test images. This preserves each
            # category's anomaly prevalence and defect-type frequencies.
            query_pool = [
                sample
                for kind in ('normal', 'abnormal')
                for species in sorted(self.metadata[sample_product][kind])
                for sample in self.metadata[sample_product][kind][species]
            ]
        else:
            query_specie_type = (
                support_normal_specie_type if query_type == 'normal'
                else support_abnormal_specie_type
            )
            query_pool = self.metadata[sample_product][query_type][query_specie_type]
        _, query_data = self.random_sample(query_pool, 1, rng)

        # Exclude the query by path. The old recursive check discarded the
        # recursive return value, allowing query/support leakage.
        query_path = query_data[0]['img_path']
        reference_pool = (self.train_normal_supports[sample_product]
                          if getattr(self, 'normal_support_split', 'test') == 'train'
                          else self.metadata[sample_product]['normal'][support_normal_specie_type])
        normal_pool = [
            sample for sample in reference_pool
            if sample['img_path'] != query_path
        ]
        abnormal_pool = [
            sample for sample in self.metadata[sample_product]['abnormal'][support_abnormal_specie_type]
            if sample['img_path'] != query_path
        ]
        _, support_normal = self.random_sample(normal_pool, self.n_shot, rng)
        _, support_abnormal = self.random_sample(abnormal_pool, self.a_shot, rng)

        # Get query image
        query_tuple = self.read_data(query_data)

        # Get normal support images
        support_normal_tuple = (0, 0, 0)
        if self.n_shot > 0:
            support_normal_tuple = self.read_data(support_normal)

        # Get abnormal support images
        support_abnormal_tuple = (0, 0, 0)
        if self.a_shot > 0:
            support_abnormal_tuple = self.read_data(support_abnormal)

        return query_tuple, support_normal_tuple, support_abnormal_tuple, sample_product


class ExhaustiveEvalDataset(FSDataset):
    """Evaluate each target test image once with fixed, disjoint train supports.

    Unlike episodic validation, this preserves the test-set class prevalence.
    The same support images are used for all queries of a product at a given
    seed, as is customary for a K-shot reference set.
    """

    def __init__(self, *args, **kwargs):
        # Opt in only with deterministic evaluation preprocessing. Each loader
        # worker owns a small cache of fixed supports; query images are not cached.
        self.cache_supports = kwargs.pop('cache_supports', False)
        self._transformed_supports = {}
        if kwargs.get('split', 'eval') not in ('eval', 'test'):
            raise ValueError('ExhaustiveEvalDataset is for evaluation only')
        kwargs['split'] = 'eval'
        super().__init__(*args, **kwargs)
        if self.a_shot:
            raise ValueError('Exhaustive evaluation supports normal references only')

        data = ['mvtec', 'visa'] if self.data_mode == 'mvtec_visa' else ['Real-IAD/realiad_1024_unzip']
        train_samples = {}
        for directory in data:
            data_path = os.path.join(self.data_root, directory)
            with open(os.path.join(data_path, self.data_name_json)) as file:
                train_info = json.load(file)['train']
            for product, samples in train_info.items():
                train_samples[product] = [
                    {'img_path': os.path.join(data_path, sample['img_path']),
                     'mask_path': os.path.join(data_path, sample['mask_path']),
                     'anomaly': sample['anomaly']}
                    for sample in samples if not sample['anomaly']
                ]

        self.query_samples = []
        self.fixed_supports = {}
        for class_id in self.class_ids:
            product = self.all_product[class_id]
            for kind in ('normal', 'abnormal'):
                for species in sorted(self.metadata[product][kind]):
                    for sample in self.metadata[product][kind][species]:
                        self.query_samples.append((product, sample))
            pool = sorted(train_samples[product], key=lambda item: item['img_path'])
            if len(pool) < self.n_shot:
                raise ValueError(f'{product}: only {len(pool)} train normals for {self.n_shot}-shot')
            rng = random.Random(f'{self.seed}:{product}')
            # Shuffle the entire pool independently of K so the 1/2/4-shot
            # reference sets are nested, rather than changing all references.
            rng.shuffle(pool)
            self.fixed_supports[product] = pool[:self.n_shot]
        self.evaluation_manifest = self.audit_manifest()

    def audit_manifest(self):
        """Fail on duplicate/leaked/missing inputs; persist exact coverage."""
        query_paths = [os.path.realpath(sample['img_path'])
                       for _, sample in self.query_samples]
        if len(query_paths) != len(set(query_paths)):
            raise ValueError('Exhaustive test metadata contains duplicate query paths')
        query_set = set(query_paths)
        references = {}
        counts = {}
        for product, samples in self.fixed_supports.items():
            paths = [os.path.realpath(sample['img_path']) for sample in samples]
            if len(paths) != len(set(paths)) or any(sample['anomaly'] for sample in samples):
                raise ValueError(f'{product}: references must be distinct normal images')
            if query_set.intersection(paths):
                raise ValueError(f'{product}: train reference overlaps with a test query')
            references[product] = paths
        for product, sample in self.query_samples:
            count = counts.setdefault(product, {'total': 0, 'normal': 0, 'anomaly': 0})
            count['total'] += 1
            count['anomaly' if sample['anomaly'] else 'normal'] += 1
            if sample['anomaly'] and not os.path.isfile(sample['mask_path']):
                raise FileNotFoundError(sample['mask_path'])
        for path in query_paths + [p for paths in references.values() for p in paths]:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
        return {
            'protocol': 'exhaustive', 'query_split': 'test', 'support_split': 'train_normal',
            'seed': self.seed, 'n_shot': self.n_shot,
            'number_of_queries': len(query_paths), 'number_of_unique_queries': len(query_set),
            'query_support_overlap': 0, 'support_selection': 'seeded_full_pool_shuffle_prefix',
            'query_paths_sha256': hashlib.sha256('\n'.join(query_paths).encode()).hexdigest(),
            'per_category_counts': counts, 'support_paths': references,
            'queries': [{'product': product, 'path': path, 'anomaly': int(sample['anomaly'])}
                        for (product, sample), path in zip(self.query_samples, query_paths)],
        }

    def __len__(self):
        return len(self.query_samples)

    def __getitem__(self, idx):
        if not self.cache_supports:
            return super().__getitem__(idx)
        product, query = self.query_samples[idx]
        query_tuple = self.read_data([query])
        query_image = self.image_list_transform(query_tuple[0])
        query_mask = self.mask_list_transform(query_tuple[1], getattr(self, 'mask_size', None) or query_image.shape[-2:])
        if product not in self._transformed_supports:
            support_tuple = self.read_data(self.fixed_supports[product])
            support_images = self.image_list_transform(support_tuple[0])
            support_masks = self.mask_list_transform(support_tuple[1], support_images.shape[-2:])
            self._transformed_supports[product] = [support_images, support_masks]
        return {
            'query': [query_image, query_mask],
            'image_level_label': torch.tensor(query_tuple[2][0], dtype=torch.long),
            'support_normal': self._transformed_supports[product],
            'support_abnormal': [0, 0],
            'sample_product': product,
        }

    def load_frame(self, idx):
        product, query = self.query_samples[idx]
        query_tuple = self.read_data([query])
        support_tuple = self.read_data(self.fixed_supports[product])
        return query_tuple, support_tuple, (0, 0, 0), product
