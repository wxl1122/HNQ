### Dataset Preparation
1. Please download MVTecAD dataset from [MVTecAD dataset](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/), VisA dataset from [VisA dataset](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar), BTAD dataset from [BTAD dataset](http://avires.dimi.uniud.it/papers/btad/btad.zip), and MVTec3D dataset from [MVTec3D dataset](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad), MPDD dataset from [MPDD dataset](https://github.com/stepanje/MPDD), MVTecLOCO dataset from [MVTecLOCO dataset](https://www.mvtec.com/company/research/datasets/mvtec-loco), BraTS dataset from [BraTS dataset](https://www.kaggle.com/datasets/dschettler8854/brats-2021-task1), RealIAD from [RealIAD dataset](https://realiad4ad.github.io/Real-IAD/), and etc. 
2. Convert the dataset to MVTecAD structure, the folder structure should be like:
   ```
   /path/to/dataset
    └── dataset_name_1
        ├── class_1
        │   ├── train
        │   │   ├── 000.png
        │   │   ├── 001.png
        │   │   └── ...
        │   ├── test
        │   │   ├── good
        │   │   │   ├── 000.png
        │   │   │   ├── 001.png
        │   │   │   └── ...
        │   │   ├── anomaly_type_1
        │   │   │   ├── 000.png
        │   │   │   ├── 001.png
        │   │   │   └── ...
        │   │   └── ...
        │   └── ground_truth
        │       ├── anomaly_type_1
        │       │   ├── 000.png
        │       │   ├── 001.png
        │       │   └── ...
        │       └── ...
        ├── class_2
        └── ...
    └── dataset_name_2
    └── ...
   ```
   For example, covert VisA dataset to MVTecAD structure by running (details in [this link](https://github.com/amazon-science/spot-diff?tab=readme-ov-file#data-preparation)):
   ```bash
   python dataset_preparation/convert_visa_to_mvtec.py
   ```



```

## Acknowledgement
We thank the great works [InCTRL](https://github.com/mala-lab/InCTRL), [ResAD](https://github.com/xcyao00/ResAD), [AnomalyDINO](https://github.com/dammsi/AnomalyDINO), [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP), and [WinCLIP](https://github.com/zqhang/Accurate-WinCLIP-pytorch) for assisting with our work.

## License
The code and dataset in this repository are licensed under the [MIT license](https://mit-license.org/).
```
