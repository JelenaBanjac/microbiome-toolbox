import setuptools
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

def parse_requirements(requirement_file, depth=0):
    """Parse requirements from a pip requirements file.

    We support:
        somepackage1
        somepackage2 => 2.3.4  # Some comments
        somepackage3 [extrastuff,otherextra] => 2.3.4  # Some comments
        -r ...

    We ignore:
        -e ...

    We do not support (and will fail if we find):
        somepackage ; some constraints
        -other options
    """
    if depth > 100:
        raise ('Detected circular dependency with "-r" in %s' % requirement_file)
    with open(requirement_file) as f:
        requirements = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-e "):
                continue
            if line.startswith("-r "):
                # Depth is used to check circular dependencies.
                requirements += parse_requirements(line.split()[1], depth + 1)
                continue

            # We take the underlined stuff and remove the spaces.
            # If a requirement line does not match this, we throw an error.
            m = re.match(
                r"^([a-zA-Z]\S*)\s*(\[[^]]+\])?\s*([=<>]+\s*\d\S*)?\s*(?:#.*)?$",
                line,
            )
            if not m:
                raise Exception("Could not parse line: {line}")
            else:
                requirements.append("".join([grp for grp in m.groups() if grp]))

    return requirements

setuptools.setup(
    name="microbiome-toolbox", 
    version="1.0.2",
    author="Jelena Banjac, Shaillay Kumar Dogra, Norbert Sprenger",
    author_email="msjelenabanjac@gmail.com, ShaillayKumar.Dogra@rd.nestle.com, norbert.sprenger@rdls.nestle.com",
    maintainer="Jelena Banjac",
    maintainer_email="msjelenabanjac@gmail.com",
    description="Microbiome Toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JelenaBanjac/microbiome-toolbox",
    packages=setuptools.find_packages(include=['microbiome', 'microbiome.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=parse_requirements("requirements.txt"),
)
