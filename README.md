# RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis
This is the benchmark code for the "RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis".  
Dataset URL: <https://huggingface.co/datasets/TokyoTechMagicYang/RAM-W600>.

## Update

- **[2025-09-19]** :tada::tada:Our dataset and benchmark paper has been accepted by NeurIPS 2025! :tada::tada:

- **[2025-08-29]** We conducted a major update of the RAM-W600 dataset.  
  In this update, we expanded the dataset with 427 additional X-ray images, updated the split and released the metadata.

- **[2025-05-10]** We released the first update of the RAM-W600 dataset, which includes 621 X-ray images.  

## Setup
- Install the conda environment
```bash
conda create -n ramw600 python=3.10
conda activate ramw600
```

- Install PyTorch
```bash
# CUDA 12.6
pip3 install torch torchvision torchaudio
```

- Install other requirements
```bash
pip install -r requirements.txt
```

## Dataset
Please refer to the link above to download the dataset.

## Run
- **Training**  
  The training configurations for segmentation and classification tasks are in `./train_seg.py` and `./train_cls.py`.  
  We also provide scripts in `./train_seg.sh` and `./train_cls.sh`.  
  Before running, you should refer to `main_seg.py` and `main_cls.py` and add your paths to the bash files.  
  After running, the checkpoints will be saved in `./ckpts/`.

```bash
bash train_seg.sh
```

- **Testing**  
  The testing configurations for segmentation and classification tasks are in `./test_seg.py` and `./test_cls.py`.  
  We also provide scripts in `./test_seg.sh` and `./test_cls.sh`.  
  After running, the results of the visualization will be saved in the folder you chose for testing.

```bash
bash test_seg.sh
```

## Citation

If you use RAM-W600 in your research, please cite:

```bibtex
@article{yang2026ram,
  title={RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis},
  author={Yang, Songxiao and Wang, Haolin and Fu, Yao and Tian, Ye and Kamishima, Tamotsu and Ikebe, Masayuki and Ou, Yafei and Okutomi, Masatoshi},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2026}
}
```

## Suggested Citation

If you use the benchmark code or experimental settings, we also recommend citing:

```bibtex
@misc{yang2026ramh1200,
  title={RAM-H1200: A Unified Evaluation and Dataset on Hand Radiographs for Rheumatoid Arthritis},
  author={Songxiao Yang and Haolin Wang and Yao Fu and Junmu Peng and Lin Fan and Hongruixuan Chen and Jian Song and Masayuki Ikebe and Shinya Takamaeda-Yamazaki and Masatoshi Okutomi and Tamotsu Kamishima and Yafei Ou},
  year={2026},
  eprint={2605.05616},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.05616}
}
```




