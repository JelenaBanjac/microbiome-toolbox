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
