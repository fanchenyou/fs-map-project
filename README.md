### Few-Shot Multi-Agent Perception
### (FS-MAP)

This is the homepage of 
* Chenyou Fan, Junjie Hu, Jianwei Huang. "Few-Shot Multi-Agent Perception." In 29th ACM International Conference on Multimedia (ACM MM'21).

![Demo](/pics/demo.png)
![Demo_air_ground](/pics/demo_air_ground.png)

## Paper
Our paper is here <a href="https://fanchenyou.github.io/homepage/docs/fs_map_1.pdf" title="FS-MAP">FS-MAP</a>.

## Dataset
We use <a href="https://microsoft.github.io/AirSim" title="AirSim">AirSim</a> dataset to perform few-shot segmentation task.
We will provide direct download link below.

## Code of Segmentation on AirSim Dataset
- The code is modified upon <a href="https://github.com/icoz69/DeepEMD" title="DeepEMD">DeepEMD</a>. Please properly cite their excellent work if you use this code in research.
- We provide self-contained implementation in the following sectoin.

### Download dataset and our split
- Download dataset from Google Drive <a href="" title="link">link</a>
- Check our split in `configs/split_save_files.pkl`

### pre-train segmentation
~~~~
python train.py --ph=0 --is_seg=1 --pretrain_dir=results/seg/pre_train
~~~~

### Train 
~~~~
python train.py --ph=1 --is_seg=1 --pretrain_dir=results/seg/pre_train
~~~~

### Evaluation
- check results/seg/meta and find the latest checkpoint dir, to replace XXX
- set "--shot=5" to test 5-shot case
~~~~
python test.py --is_seg=1 --model_dir=XXXX  --loop=0
python test.py --is_seg=1 --model_dir=XXXX  --loop=0 --shot=5
~~~~
to use our trained models, download here, then execute
~~~~
python test.py --is_seg=1 --model_dir=results/seg/meta/loop0
python test.py --is_seg=1 --model_dir=results/seg/meta/loop0_st5 --shot=5
~~~~

## Reference
Please cite our work if you use this code.
~~~~
@inproceedings{fan2021fsmap,
  title={Few-Shot Multi-Agent Perception},
  author={Fan, Chenyou and Hu, Junjie and Huang, Jianwei},
  booktitle={ACM MultiMedia},
  year={2021}
}
~~~~

## Other references
Please also properly cite the following excellent work in research.
- <a href="https://github.com/icoz69/DeepEMD" title="DeepEMD">DeepEMD</a>. 
- <a href="https://ycliu93.github.io/projects/multi-agent-perception.html" title="MAP">Multi-Agent Perception</a>. 




## Requirements
Python = 3.8
PyTorch = 1.7+ [[here]](https://pytorch.org/)

GPU training with 4G+ memory, testing with 2G+ memory.

~~~~
pip install scikit-learn pretrainedmodels
~~~~