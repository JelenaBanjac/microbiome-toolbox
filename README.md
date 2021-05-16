<div align="center">
  <p>
  <img src="https://image.freepik.com/free-vector/pathogen-microorganisms-set_74855-5909.jpg" alt="https://www.freepik.com/pch-vector" width="270" />
  </p>
  <p>
    <a href="">
      <img alt="First release" src="https://img.shields.io/badge/release-v1.0-brightgreen.svg" />
    </a>
  </p>
  
  <p>
    <a href="https://microbiome-toolbox.herokuapp.com">
      Dashboard
    </a>
  </p>
</div>


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
```
# create environment
conda env create -f environment.yml

# activate environment
conda activate microbiome

# install microbiome toolbox
pip install microbiome-toolbox --user
```

## Examples

For the toolbox usage, checkout the notebooks:
- [1. Mouse Data Extraction (example data)](https://nbviewer.jupyter.org/github/JelenaBanjac/microbiome-toolbox/blob/main/notebooks/Mouse_16S/mousedata_test.ipynb)
- [2. Data Analysis and Exploration](https://nbviewer.jupyter.org/github/JelenaBanjac/microbiome-toolbox/blob/main/notebooks/Mouse_16S/mouse_16S_healthy_reference_definition.ipynb)
- [3. Microbiome trajectory and Outlier Detection with Intervention Simulation](https://nbviewer.jupyter.org/github/JelenaBanjac/microbiome-toolbox/blob/main/notebooks/Mouse_16S/mouse_analysis_16S.ipynb)

## Licence 
The project is licensed under the [MIT license](./LICENCE).

## Authors
[Jelena Banjac](msjelenabanjac@gmail.com), [Shaillay Kumar Dogra](ShaillayKumar.Dogra@rd.nestle.com), [Norbert Sprenger](norbert.sprenger@rdls.nestle.com)
