# RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis
This is the benchmark code for the "RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis".  
Dataset URL: <https://huggingface.co/datasets/TokyoTechMagicYang/RAM-W600>.

## Update
- **[2025-05-10]** We released the first update of the RAM-W600 dataset, which includes 621 X-ray images.  

- **[2025-08-29]** We conducted a major update of the RAM-W600 dataset.  
  In this update, we expanded the dataset with 427 additional X-ray images and released part of the metadata.  
  

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
