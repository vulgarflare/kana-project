# Kana Project: Semi-Supervised Interpretation of Colorimetric Paper-Based Sensor Images

This repository contains the code, notebooks, and experiment pipeline for our study on semi-supervised interpretation of colorimetric paper-based sensor images for on-site kanamycin monitoring.

Our work is built on top of the [USB: Unified Semi-Supervised Learning Benchmark](https://github.com/microsoft/Semi-supervised-learning) framework, with substantial modifications for low-sample colorimetric sensing data. In particular, we adapted FixMatch to better fit the characteristics of our small-scale colorimetric image dataset and introduced a **Shape Consistency Prior (SCP)** module.

## Project Overview

Monitoring antibiotic contamination in water requires rapid, low-cost, and field-deployable methods. In this project, we combine:

- an AuNP-based paper colorimetric sensor for kanamycin detection,
- standardized smartphone image acquisition with a 3D-printed dark box,
- a semi-supervised learning framework for image-based concentration classification.

The resulting dataset, **Kana**, is a small-sample colorimetric image dataset with three concentration classes and strong color-dependent discrimination.

## Main Contributions

- Built a small-sample colorimetric image dataset for kanamycin detection.
- Adapted the FixMatch semi-supervised learning framework for this domain.
- Proposed a **Shape Consistency Prior (SCP)** module to inject domain knowledge into SSL training.
- Designed overlay-based augmentation strategies for colorimetric reaction region guidance.
- Evaluated the method under different labeled-data regimes using ResNet and ViT backbones.

## Repository Structure

```text
.
├── notebooks/
│   ├── ...                         # Experiment notebooks and usage examples
├── semilearn/
│   └── algorithms/
│       ├── fixmatch_Overlay2Img/
│       ├── fixmatch_Shape_Consistency_Loss/
│       └── fixmatch_Shape_Consistency_Loss_Overlay2Img/
├── README.md
└── ...
````

## Key Modified Components

The main modifications in this project are located in:

* `notebooks/`
* `semilearn/algorithms/fixmatch_Overlay2Img`
* `semilearn/algorithms/fixmatch_Shape_Consistency_Loss`
* `semilearn/algorithms/fixmatch_Shape_Consistency_Loss_Overlay2Img`

These modules implement our domain-adapted FixMatch variants and experimental workflows.

## Dataset

The Kana dataset is currently provided as an archived backup package.

**Dataset download:**

* `kana_data_backup.tar.gz`
* Quark link: [https://pan.quark.cn/s/48ad75ec4534](https://pan.quark.cn/s/48ad75ec4534)

The dataset consists of colorimetric sensor images collected under standardized acquisition conditions using a custom dark-box setup.

### Class setting

The current experiments use three concentration categories:

* Low: 0–100 nM
* Medium: 200–300 nM
* High: 400–2000 nM

### Labeled-data regimes

Experiments were conducted under four SSL settings:

* `kana_1`
* `kana_5`
* `kana_10`
* `kana_15`

where the number indicates the labeled samples per class.

## Method Summary

Our method extends FixMatch with two task-specific components:

### 1. Overlay augmentation

A soft mask derived from a blue-score response is embedded into the red channel of the image to emphasize the colorimetrically salient region.

### 2. Shape Consistency Prior (SCP) loss

A mask-weighted entropy minimization loss is introduced to encourage stable predictions under prior-guided perturbations.

Together, these components improve performance and stability in low-label training settings.

## Experimental Notes

* Base framework: USB / SemiLearn
* Main SSL baseline: FixMatch
* Backbones evaluated:

  * ResNet50
  * Vision Transformer (ViT)
* Best results in our experiments were achieved with the **ResNet backbone** under the combined **FixMatchSCP-Overlay** setting.

## Reproducibility

The notebooks in `notebooks/` document the experimental process and usage examples.

The custom algorithm implementations are available under:

* `fixmatch_Overlay2Img`
* `fixmatch_Shape_Consistency_Loss`
* `fixmatch_Shape_Consistency_Loss_Overlay2Img`

## Manuscript Status

This repository supports an unpublished manuscript currently under preparation:

**Semi-Supervised Interpretation of Colorimetric Paper-Based Sensor Images with a Shape Consistency Prior for On-Site Antibiotic Monitoring**

Please note that the manuscript is not yet formally published.

## Acknowledgement

This project is developed based on the USB / SemiLearn framework:

* USB: A Unified Semi-Supervised Learning Benchmark for Classification

We thank the original authors for releasing their excellent codebase and benchmark.

## Citation

If you use this repository, please cite the related work after formal publication.

```bibtex
@misc{kana_project,
  title={Semi-Supervised Interpretation of Colorimetric Paper-Based Sensor Images with a Shape Consistency Prior for On-Site Antibiotic Monitoring},
  author={Ningxin Zhang and Xiaoyu Zhang and Xiangyu Chen and Qiao Cao and Rujing Wang},
  year={2026},
  note={Manuscript in preparation}
}
```

## Contact

For academic communication regarding this project, please contact the corresponding author listed in the manuscript.
