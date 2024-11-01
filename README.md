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

We introduce a grouped dropout strategy and modify the CNN architecture to improve the accuracy of multi-class insect recognition. Leveraging the Inception module’s branching structure and the adaptive grouping properties of the WeDIV clustering algorithm, we developed two grouped dropout models, the iGDnet-IP and
GDnet-IP. Experimental results on a dataset containing 20 insect species (15 pests and five beneficial insects) with 73,635 images demonstrated an increase in cross-validation accuracy from 84.68% to 92.12%, with notable improvements in the recognition rates for difficult-to-classify species. Our model showed significant accuracy advantages over standard dropout methods on independent test sets, with much less training time compared to four conventional CNN models, highlighting the suitability for mobile applications.

<p style="float: center">
  <img src="images/arch-GDnet-IP.svg" alt="Architecture of GDnet-IP" />
  <div align="center">(A)</div>
</p>
<p float="center">
  <img src="images/iGDnet-IP.svg" style="display: block; width: 350px; height: auto" />
  <img src="images/weGDnet-IP.svg" style="display: block; width: 450px; height: auto" />
</p>
<div>
  <span style="font-weight: bold;">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(B)</span> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
  <span style="font-weight: bold;">(C)</span>
</div>

<p align="left"><b>Grouped dropout-based CNN for insect pest recognition. (A) Architecture of GDnet-IP; (B) Inception-based GDnet-IP, where the grey branch is randomly deactivated; (C) Clustering-based GDnet-IP, where the channels in 'Group 2' are randomly deactivated.</b></p>

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
