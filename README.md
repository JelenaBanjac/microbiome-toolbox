<div align="center">
  <p>
  <img src="https://media.giphy.com/media/ZutFV1eVDiXUlbBT5B/giphy.gif" width="220" />
  </p>
  <p>
    <a href="">
      <img alt="First release" src="https://img.shields.io/badge/release-v1.0-brightgreen.svg" />
    </a>
  </p>

  <p>
    <strong>Early Life Microbiome Toolbox</strong>
  </p>
  
  <p>
    <a href="https://TODO.github.io">
      Website
    </a>
  </p>
</div>


# Early Life Microbiome Toolbox

**Note:**
```bash
# building package
#python setup.py sdist bdist_wheel
python setup.py develop
```

## Songbird
#songbird==0.8.2
```
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