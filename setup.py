from setuptools import setup
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'nndcp',
    version = '0.0.1',
    description = 'neural network training with difference of convex programming (DCP)',
    author = "Yifei Liu",
    author_email = "liu00980@umn.edu",
    url = "https://github.umn.edu/liu00980",
    keywords = ["deep learning", "optimization", "neural network", "difference of convex programming"],
    py_modules = ["DCshallow", "SGDtraining", "data"],
    package_dir = { '': 'src'},
    include_package_data = True,
    package_data = {
        "": ["realdata/*.pkl"],
        "": ["realdata/*.npy"]
    },
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    long_description = long_description,
    long_description_content_type = "text/markdown",
    install_requires = [
        'matplotlib',
        'pandas',
        'numpy',
        'matplotlib',
        'cvxpy',
        'scs',
        'mosek',
        'scipy',
        'sklearn',
        'torch',
    ],
	extras_require = {
        "dev": [
            "pytest >= 3.7",
        ],
    },
)