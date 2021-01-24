# AIDE
AIDE: Annotation-efficient deep learning for automatic medical image segmentation

## Quick start
### Install
1. Install PyTorch=1.1.0 following the [official instructions](https://pytorch.org/).
2. git clone [https://github.com/lich0031/AIDE](https://github.com/lich0031/AIDE).
3. Install dependencies: pip install -r requirements.txt

## Data preparation
- If you want to run the code, you need to download the [CHAOS](https://chaos.grand-challenge.org/), [Prostate dataset1](https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures), [Prostate dataset2](https://promise12.grand-challenge.org/Home/), and [QUBIQ](https://qubiq.grand-challenge.org/) datasets for respective tasks.

- Example optimized models and segmentation results for the task utilizing CHAOS dataset can be downloaded [here](https://onedrive.live.com/?id=D6A80DBCD21AD447%216335&cid=D6A80DBCD21AD447).

## Train the model
- Please specify the configuration file.
- For example, train the model on CHAOS with a batch size of 4 on GPU 0:

  python train_files/trainchaos_comparison_1case.py --model_name fuseunet --batch_size 4 --gpu_order 0 --repetition 1
