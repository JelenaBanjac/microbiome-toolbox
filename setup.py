import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# TODO: authors and emails, and url
setuptools.setup(
    name="microbiome-toolbox", 
    version="0.0.11",
    author="Jelena Banjac, Shaillay Kumar Dogra, Norbert Sprenger",
    author_email="msjelenabanjac@gmail.com, ShaillayKumar.Dogra@rd.nestle.com, norbert.sprenger@rdls.nestle.com",
    maintainer="Jelena Banjac",
    maintainer_email="msjelenabanjac@gmail.com",
    description="Microbiome Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(include=['microbiome', 'microbiome.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "seaborn",
        "ipyvolume",
        "xgboost",
        "celery",
        "redis",
        #"ca-certificates==2020.10.14",
        "certifi==2020.6.20",
        "cycler==0.10.0",
        #"freetype==2.8",
        #"icu==58.2",
        #"jpeg==9b",
        #"kiwisolver==1.2.0",
        #"libpng==1.6.37",
        #"libxgboost==0.90",
        "matplotlib==2.2.2",
        "pyparsing==2.4.7",
        # "sip==4.19.24",
        "backcall==0.2.0",
        "colorama==0.4.4",
        "decorator==4.4.2",
        "icc_rt==2019.0.0",
        "ipykernel==5.3.4",
        "ipython==7.16.1",
        "ipython_genutils==0.2.0",
        "jedi==0.17.2",
        "joblib==0.17.0",
        "jupyter_client==6.1.7",
        "jupyter_core==4.6.3",
        # "mkl-service==2.3.0",
        # "mkl_fft==1.2.0",
        # "mkl_random==1.1.1",
        "numpy==1.19.1",
        "pandas==1.1.3",
        "parso==0.7.0",
        "pickleshare==0.7.5",
        "pip==20.2.3",
        "prompt-toolkit==3.0.8",
        "pygments>=2.7.1",
        "python-dateutil==2.8.1",
        # "pytz==2020.1",
        # "pywin32==227",
        # "pyzmq==19.0.2",
        "scikit-learn==0.23.2",
        #"scikit-bio",
        "scipy==1.5.2",
        "setuptools==50.3.0",
        "six==1.15.0",
        "threadpoolctl==2.1.0",
        "tornado",
        "traitlets==4.3.3",
        # "wcwidth==0.2.5",
        "wheel==0.35.1",
        # "wincertstore==0.2",
        "dash-bootstrap-components",
        "dash-html-components",
        "dash-core-components",
        "dash",
        "dash-renderer",
        "dash-table",
        "dash-uploader",
        "plotly",
        "flask-caching",
        "catboost",
        "statsmodels",
        "numpy",
        #"- scikit-bio",
        "shap",
        "tk",
        "dash-dangerously-set-inner-html",
        "gunicorn"
    ]
)
