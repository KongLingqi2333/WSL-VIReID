# Official code of ICCV 2025 Paper "[Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](https://arxiv.org/abs/2507.12942)"
## Step 1 Datasets Preparation
* [SYSU-MM01](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_RGB-Infrared_Cross-Modality_Person_ICCV_2017_paper.pdf)
```python
python pre_process_sysu.py
```
* [LLCM](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Diverse_Embedding_Expansion_Network_and_Low-Light_Cross-Modality_Benchmark_for_Visible-Infrared_CVPR_2023_paper.pdf)
* [RegDB](https://www.mdpi.com/1424-8220/17/3/605)
## Step 2 Requirements
```python
python -m pip install -r requirements.txt
```
## Step 3 Training
```bash
# SYSU-MM01
sh sysu.sh
# LLCM
sh llcm.sh
# RegDB
sh regdb.sh
```

## Citation
```
@misc{zhang2025weaklysupervisedvisibleinfraredperson,
      title={Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning}, 
      author={Yafei Zhang and Lingqi Kong and Huafeng Li and Jie Wen},
      year={2025},
      eprint={2507.12942},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.12942}, 
}
```
