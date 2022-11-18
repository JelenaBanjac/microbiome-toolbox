<div align="center">
  <p>
  <img src="https://www.insideprecisionmedicine.com/wp-content/uploads/2019/07/Jul1_2019_GettyImages_927406508_BacterialMicrooginism-1.jpg" alt="Alexander Mikhailov / iStock / Getty Images" width="270" />
  </p>
  <p>
    <a href="">
      <img alt="First release" src="https://img.shields.io/badge/release-v1.0-brightgreen.svg" />
    </a>
  </p>
  
  <p>
    <a href="https://microbiome-toolbox.azurewebsites.net/">
      Dashboard
    </a>
  </p>
</div>

- [Dashboard](https://microbiome-toolbox.azurewebsites.net/) with interactive visualizations
- [BioRxiv](https://www.biorxiv.org/content/10.1101/2022.02.14.479826v1) paper (preprint)
- We are on the [curated set](https://dash-demo.plotly.host/Portal/) and [Plotly&Dash 500](https://dash-demo.plotly.host/plotly-dash-500/) of STEM focused Plotly Dash apps (as Microbiome-Toolbox)!

# Microbiome Toolbox

Microbiome toolbox is a collection of tools and methods for microbiome data and it includes data analysis and exploration, data preparation, microbiome trajectory modeling, outlier discovery and intervention. Our toolbox encompasses most of the common machine learning algorithms that exist in different packages.

Features:
- Data analysis and exploration of microbiota data
- Data preparation
- Reference vs. non-reference data analysis
- Log-ratios data transformation
- Microbiome trajectory
- Boxplot with time
- Intervention simulation.



## Installation
The microbiome toolbox has a [PyPi package](https://pypi.org/project/microbiome-toolbox/) available.

```bash
# create environment
conda env create -f environment.yml

# activate environment
conda activate microbiome

# install microbiome toolbox
pip install microbiome-toolbox --user
```

## Run dashboard locally (on your computer)
After you successfully installed the microbiome-toolbox and activated the environment, just execute the following commands:

```bash
# set up the development environment (on Linux)
source webapp/environment/.evv.development

# run the server
python webapp/index.py
```
The only step that differs for Windows is that you should modify the environment variables with values indicated in file `webapp/environment/.evv.development`. 
After that, you can run the server on Windows.

## Examples

For the toolbox usage, checkout the notebooks:
- [1. Mouse dataset](https://nbviewer.org/github/JelenaBanjac/microbiome-toolbox/tree/main/notebooks/Mouse_16S/microbiome_dataset.ipynb)
- [2. Mouse trajectory](https://nbviewer.org/github/JelenaBanjac/microbiome-toolbox/blob/main/notebooks/Mouse_16S/microbiome_trajectory.ipynb)
- [3. Human infants dataset](https://nbviewer.org/github/JelenaBanjac/microbiome-toolbox/tree/main/notebooks/Human_Subramanian/microbiome_dataset.ipynb)
- [4. Human infants trajectory](https://nbviewer.org/github/JelenaBanjac/microbiome-toolbox/blob/main/notebooks/Human_Subramanian/microbiome_trajectory.ipynb)

## Issues
If you notice any issues, please report them at [Github issues](https://github.com/JelenaBanjac/microbiome-toolbox/issues).

## Licence 
The project is licensed under the [MIT license](./LICENCE).

## Authors
[Jelena Banjac](https://jelenabanjac.com/), [Shaillay Kumar Dogra](ShaillayKumar.Dogra@rd.nestle.com), [Norbert Sprenger](norbert.sprenger@rdls.nestle.com)

## Citation
The code in this repository is released under the terms of the [MIT license](./LICENCE.md). Please cite our paper if you use it.

BibTeX citation style:
```
@article{microbiome_toolbox,
  author = {Banjac, Jelena and Sprenger, Norbert and Dogra, Shaillay Kumar},
  title = {Microbiome Toolbox: Methodological approaches to derive and visualize microbiome trajectories},
  elocation-id = {2022.02.14.479826},
  year = {2022},
  doi = {10.1101/2022.02.14.479826},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2022/02/16/2022.02.14.479826},
  eprint = {https://www.biorxiv.org/content/early/2022/02/16/2022.02.14.479826.full.pdf},
  journal = {bioRxiv}
}
```
APA citation style:
```
Banjac J, Sprenger N, Dogra SK. 2022. Microbiome Toolbox: Methodological approaches to derive and visualize microbiome trajectories. bioRxiv doi: 10.1101/2022.02.14.479826
https://biorxiv.org/cgi/content/short/2022.02.14.479826v1
```
