# IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization and Perturbation Optimization.

🧑‍🔬 Authors: Subrat Kishore Dutta, Xiao Zhang

🧠 [AIRML Lab](https://air-ml.org/)

📚 Publication: [Arxiv](https://openaccess.thecvf.com/content/ICCV2025/papers/Dutta_IAP_Invisible_Adversarial_Patch_Attack_through_Perceptibility-Aware_Localization_and_Perturbation_ICCV_2025_paper.pdf)

---
<img src="/assets/flowchart.jpg" alt="IAP Teaser" width="850"/>

## Abstract:

<p>
        Despite modifying only a small localized input region, adversarial patches can drastically change the prediction of
        computer vision models. However, prior methods either cannot perform satisfactorily under targeted attack scenarios
        or fail to produce contextually coherent adversarial patches,
        causing them to be easily noticeable by human examiners
        and insufficiently stealthy against automatic patch defenses.
        In this paper, we introduce IAP, a novel attack framework
        that generates highly invisible adversarial patches based on
        perceptibility-aware localization and perturbation optimization schemes. Specifically, IAP first searches for a proper
        location to place the patch by leveraging classwise localization and sensitivity maps, balancing the susceptibility of
        patch location to both victim model prediction and human
        visual system, then employs a perceptibility-regularized adversarial loss and a gradient update rule that prioritizes
        color constancy for optimizing invisible perturbations. Comprehensive experiments across various image benchmarks
        and model architectures demonstrate that IAP consistently
        achieves competitive attack success rates in targeted settings
        with significantly improved patch invisibility compared to
        existing baselines. In addition to being highly imperceptible
        to humans, IAP is shown to be stealthy enough to render
        several state-of-the-art patch defenses ineffective.
</p>

---

## Highlights:

<ul>
        <li>We propose a novel imperceptible adversarial patch attack method using perceptibility-aware optimization.</li>
        <li>We acheive state-of-the-art attack efficacy and imperceptibility on ImageNet and VGGFace dataset</li>
        <li>We successfully bypass multiple state-of-the-art adversarial patch defense methods</li>
      </ul>

---

## ⚙️ Setup

### Create Environment

```bash
conda create -n iap python=3.10 -y
conda activate iap
pip install -r requirements.txt
```

### Directory Structure

The repository is organized as follows. The dataset directory should be organized as shown under ../imagenet1000main/, where five images are used per ImageNet class, and one image is randomly selected for the attack.
To compute attention maps using **Grad-CAM**, we use the official PyTorch implementation provided by  
[Jacob Gildenblat’s pytorch-grad-cam repository](https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam).

```text
IAP/
├── imagenet1000main/
│   ├── 0/ 
│   │   ├── 0.jpeg
│   │   ├── 1.jpeg
│   │   └── ...
│   ├── 1/
│   └── ...
├── src/
│   ├── IAP.py
│   ├── utils.py
│   └── pytorch_grad_cam/
├── assets/
│   └── flowchart.jpg
├── results/
|   ├──data #contains the per sample attack details
|   ├──temp #contains the original version of the image at the local level
|   ├──delta #contains the final perturbed version of the image at the local level 
|   ├──imagetem #contains the original unperturbed sample
|   └──pur #contains the final adversarial sample
├── README.md
├── LICENSE
└── requirements.txt
```

---

## 💡 Running IAP

To run the attack, using the default hyperparameters reported in the paper, run:

```
python src/IAP.py
```
The resultant adversarial sample, along with other details, will be stored at "results/".

## Citation
If you use this work, please cite:

```
@inproceedings{dutta2025iap,
  title={IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization and Perturbation Optimization},
  author={Dutta, Subrat Kishore and Zhang, Xiao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14766--14775},
  year={2025}
}
```
