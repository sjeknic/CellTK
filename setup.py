import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="celltk",
    version="0.3.1",
    author="Stevan Jeknic",
    author_email="sjeknic@stanford.edu",
    description="A tool kit for working with large amounts of live-cell microscopy data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sjeknic/CellTK",
    project_urls={
        "Bug Tracker": "https://github.com/sjeknic/CellTK/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "celltk"},
    packages=setuptools.find_packages(where="celltk"),
    install_requires=[],
    python_requires=">=3.8",
)