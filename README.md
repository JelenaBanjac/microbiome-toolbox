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
    <strong>Microbiome Toolbox</strong>
  </p>
  
  <p>
    <a href="https://microbiome-toolbox.herokuapp.com">
      Website
    </a>
  </p>
</div>


# Microbiome Toolbox

!pip install microbiome-toolbox==0.0.4 --user

Install your package from the real PyPI using python3 -m pip install [your-package].

**Note:**
```bash
# building package
#python setup.py sdist bdist_wheel
python setup.py develop
```

## Songbird

```
source activate microbiome

# These instructions are identical to the Linux (64-bit) instructions (or download manually from the link)
wget https://data.qiime2.org/distro/core/qiime2-2020.11-py36-linux-conda.yml
conda env create -n qiime2-2020.11 --file qiime2-2020.11-py36-linux-conda.yml
# OPTIONAL CLEANUP
rm qiime2-2020.11-py36-linux-conda.yml

pip install songbird==0.8.2

songbird multinomial \
	--input-biom ../../temp2/data/oral-collapsed-table.biom \
	--metadata-file ../../temp2/data/oral_trimmed_metadata.txt \
	--formula "C(label, Treatment('other'))" \
	--epochs 10000 \
	--differential-prior 0.5 \
	--training-column Test \
	--summary-interval 1 \
	--summary-dir results
```

```
songbird multinomial --input-biom ../../temp2/data/oral-collapsed-table.biom --metadata-file ../../temp2/data/oral_trimmed_metadata.txt --formula "C(label, Treatment('other'))" --epochs 10000 --differential-prior 0.5 --training-column Test --summary-interval 1 --summary-dir results
```
