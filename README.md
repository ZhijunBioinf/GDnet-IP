<a id="readme-top"></a>

[downloads]: https://img.shields.io/github/downloads/ZhijunBioinf/GDnet-IP/total.svg?style=social&logo=github&label=Download
[downloads-url]: https://github.com/ZhijunBioinf/GDnet-IP/releases
[stars-shield]: https://img.shields.io/github/stars/ZhijunBioinf/GDnet-IP.svg?style=flat-square&color=red
[stars-url]: https://github.com/ZhijunBioinf/GDnet-IP/stargazers
[forks-shield]: https://img.shields.io/github/forks/ZhijunBioinf/GDnet-IP.svg?style=flat-square&color=blue
[forks-url]: https://github.com/ZhijunBioinf/GDnet-IP/network/members

# GDnet-IP: Grouped Dropout-Based Convolutional Neural Network for Insect Pest Recognition</h1>

[![Downloads][downloads]][downloads-url]
[![Stargazers][stars-shield]][stars-url]
[![Forks][forks-shield]][forks-url]

## DESCRIPTION

We introduce a grouped dropout strategy and modifies the CNN architecture to improve the accuracy of multi-class insect recognition. Leveraging the Inception module’s branching structure and the adaptive grouping properties of the WeDIV clustering algorithm, we developed two grouped dropout models, the iGDnet-IP and
GDnet-IP. Experimental results on a dataset containing 20 insect species (15 pests and five beneficial insects) with 73,635 images demonstrated an increase in cross-validation accuracy from 84.68% to 92.12%, with notable improvements in the recognition rates for difficult-to-classify species. Out model showed significant accuracy advantages over standard dropout methods on test sets, with faster training times compared to four conventional CNN models, highlighting the suitability for mobile applications.

<div style="position: relative; display: inline-block;">
  <img src="./gdnet-ip-arch.png" alt="Architecture of GDnet-IP" />
  <div align="center">(A)</div>
</div>

<p align="center"><b>Architecture of GDnet-IP</b></p>

## Getting Started

GDnet-IP has been tested with Python 3.?? and PyTorch 2.??. The user can easily set up the required environment using Conda by following these steps:

- Clone the repository
  
  ```bash
  git clone https://github.com/ZhijunBioinf/GDnet-IP.git
  cd GDnet-IP
  ```
  
- Create and activate Conda environment
  
  ```bash
  conda env create -f environment.yml 
  conda activate GDnet-IP
  ```
  
  <p align="right">(<a href="#readme-top">back to top</a>)</p>
  
## Usage

### Quick Start

To get started with GDnet-IP, you can load the GDnet-IP models directly by using the following Python script:

```python
from ?? import ??

# Load the GDnet-IP model
model = 
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Implementing: Dongcheng Li (dongchengli287@gmail.com)  
Supervisor: Zhijun Dai (daizhijun@hunau.edu.cn)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## [Citation](https://www.mdpi.com/2077-0472/14/11/1915)

**Dongcheng Li, Yongqi Xu, Zheming Yuan, Zhijun Dai\*. GDnet-IP: Grouped Dropout-Based Convolutional Neural Network for Insect Pest Recognition. Agriculture, 2024, 14(11), 1915.**

<p align="right">(<a href="#top">back to top</a>)</p>
