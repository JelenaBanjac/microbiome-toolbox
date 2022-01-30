import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# TODO: authors and emails, and url
setuptools.setup(
    name="microbiome-toolbox", 
    version="1.0.0",
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
        # "celery",
        # "redis",
        #"ca-certificates==2020.10.14",
        "certifi",
        "cycler",
        #"freetype==2.8",
        #"icu==58.2",
        #"jpeg==9b",
        #"kiwisolver==1.2.0",
        #"libpng==1.6.37",
        #"libxgboost==0.90",
        "matplotlib",  #==2.2.2
        "pyparsing",
        # "sip==4.19.24",
        # "backcall",
        # "colorama",
        # "decorator",
        "ipykernel",
        "ipython",
        "ipython_genutils",
        "jedi",
        "joblib",
        "jupyter_client",
        "jupyter_core",
        # "mkl-service==2.3.0",
        # "mkl_fft==1.2.0",
        # "mkl_random==1.1.1",
        "numpy",
        "pandas",
        "parso",
        "pickleshare",
        "pip",
        "prompt-toolkit",
        "pygments",
        "python-dateutil",
        # "pytz==2020.1",
        # "pywin32==227",
        # "pyzmq==19.0.2",
        "scikit-learn",
        #"scikit-bio",
        "scipy",
        "setuptools",
        "six",
        "threadpoolctl",
        "tornado",
        "traitlets",
        # "wcwidth==0.2.5",
        "wheel",
        # "wincertstore==0.2",
        # "dash-bootstrap-components",
        # "dash-html-components",
        # "dash-core-components",
        "dash",
        "diskcache",
        "multiprocess",
        "psutil",
        "dash-renderer",
        "dash-table",
        "dash-uploader",
        "dash-extensions",
        "plotly",
        "flask-caching",
        # "catboost",
        "statsmodels",
        "numpy",
        #"- scikit-bio",
        "shap",
        "tk",
        # "dash-dangerously-set-inner-html",
        "gunicorn",
        "diskcache",
        "python-dotenv",
        "psutil",
        "diskcache",
        "multiprocess",
        "natsort",
        "umap-learn",
        "black",
        "isort",
        "flake8",
    ]
)
