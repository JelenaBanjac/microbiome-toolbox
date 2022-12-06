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
- [Bioinformatics](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac781/6873738) paper
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
@article{10.1093/bioinformatics/btac781,
    author = {Banjac, Jelena and Sprenger, Norbert and Dogra, Shaillay Kumar},
    title = "{Microbiome Toolbox: Methodological approaches to derive and visualize microbiome trajectories}",
    journal = {Bioinformatics},
    year = {2022},
    month = {12},
    abstract = "{The gut microbiome changes rapidly under the influence of different factors such as age, dietary changes, or medications to name just a few. To analyze and understand such changes we present a microbiome analysis toolbox. We implemented several methods for analysis and exploration to provide interactive visualizations for easy comprehension and reporting of longitudinal microbiome data.Based on the abundance of microbiome features such as taxa as well as functional capacity modules, and with the corresponding metadata per sample, the toolbox includes methods for 1) data analysis and exploration, 2) data preparation including dataset-specific preprocessing and transformation, 3) best feature selection for log-ratio denominators, 4) two-group analysis, 5) microbiome trajectory prediction with feature importance over time, 6) spline and linear regression statistical analysis for testing universality across different groups and differentiation of two trajectories, 7) longitudinal anomaly detection on the microbiome trajectory, and 8) simulated intervention to return anomaly back to a reference trajectory.The software tools are open source and implemented in Python. For developers interested in additional functionality of the toolbox, it is modular allowing for further extension with custom methods and analysis. The code, python package, and the link to the interactive dashboard are available on GitHub https://github.com/JelenaBanjac/microbiome-toolbox.Supplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac781},
    url = {https://doi.org/10.1093/bioinformatics/btac781},
    note = {btac781},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btac781/47589252/btac781.pdf},
}

```
APA citation style:
```
Jelena Banjac, Norbert Sprenger, Shaillay Kumar Dogra, Microbiome Toolbox: Methodological approaches to derive and visualize microbiome trajectories, Bioinformatics, 2022;, btac781, 
https://doi.org/10.1093/bioinformatics/btac781
```
