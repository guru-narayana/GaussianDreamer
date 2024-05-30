
# Gdreamer

The current project is modifcation and improvement of the esisting work cited below.

## GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models (CVPR 2024)
### [Project Page](https://taoranyi.com/gaussiandreamer/) | [arxiv Paper](https://arxiv.org/abs/2310.08529)

[GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models](https://taoranyi.com/gaussiandreamer/)  

[Taoran Yi](https://github.com/taoranyi)<sup>1</sup>,
[Jiemin Fang](https://jaminfong.cn/)<sup>2‡</sup>, [Junjie Wang](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=9Nw_mKAAAAAJ)<sup>2</sup>, [Guanjun Wu](https://guanjunwu.github.io/)<sup>3</sup>,  [Lingxi Xie](http://lingxixie.com/)<sup>2</sup>, </br>[Xiaopeng Zhang](https://scholar.google.com/citations?user=Ud6aBAcAAAAJ&hl=zh-CN)<sup>2</sup>,[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Qi Tian](https://www.qitian1987.com/)<sup>2</sup> , [Xinggang Wang](https://xwcv.github.io/)<sup>1‡✉</sup>

<sup>1</sup>School of EIC, HUST &emsp;<sup>2</sup>Huawei Inc. &emsp; <sup>3</sup>School of CS, HUST &emsp; 

<sup>‡</sup>Project lead.  <sup>✉</sup>Corresponding author. 



## Get Started
**Installation**
Install [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Shap-E](https://github.com/openai/shap-e#usage) as fellow:
```
conda create -n gdreamer -y python=3.8

git clone https://github.com/hustvl/GaussianDreamer.git 

pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

pip install ninja

cd GaussianDreamer

pip install -r requirements.txt

conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
conda install conda-forge::glm

pip install ./gaussiansplatting/submodules/diff-gaussian-rasterization
pip install ./gaussiansplatting/submodules/simple-knn
pip install plyfile
pip install ipywidgets
pip install open3d

git clone https://github.com/openai/shap-e.git
cd shap-e
pip install -e .

pip install git+https://github.com/bytedance/MVDream
```
Download [finetuned Shap-E](https://huggingface.co/datasets/tiange/Cap3D/resolve/9bfbfe7910ece635e8e3077bed6adaf45186ab48/our_finetuned_models/shapE_finetuned_with_330kdata.pth) by Cap3D, and put it in `./load`

https://huggingface.co/MVDream/MVDream/tree/main
**Quickstart**

Text-to-3D Generation
```
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a fox"

# if you want to import the generated 3D assets into the Unity game engine.
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a fox" system.sh_degree=3 
```

Text-to-Avatar Generation
```
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Spiderman stands with open arms" system.load_type=1

# if you want to import the generated 3D assets into the Unity game engine.
python launch.py --config configs/gaussiandreamer-sd.yaml --train --gpu 0 system.prompt_processor.prompt="Spiderman stands with open arms" system.load_type=1 system.sh_degree=3 
```


**Application**

Import the generated 3D assets into the Unity game engine to become materials for games and designs with the help of [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting).
![block](./images/unity.gif)


## 📑 Citation
If you find this repository/work helpful in your research, welcome to cite the paper and give a ⭐.
Some source code of ours is borrowed from [Threestudio](https://github.com/threestudio-project/threestudio), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [depth-diff-gaussian-rasterization](https://github.com/ingra14m/depth-diff-gaussian-rasterization). We sincerely appreciate the excellent works of these authors.
```
@inproceedings{yi2023gaussiandreamer,
  title={GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models},
  author={Yi, Taoran and Fang, Jiemin and Wang, Junjie and Wu, Guanjun and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
  year = {2024},
  booktitle = {CVPR}
}
```
