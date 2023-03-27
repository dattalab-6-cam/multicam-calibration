import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='multicam-calibration',
    version='0.0.0',
    author='Caleb Weinreb',
    author_email='calebsw@gmail.com',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.7',
    install_requires=[
        'scipy',
        'vidio',
        'networkx',
        'tqdm',
        'matplotlib',
    ], 
    url='https://github.com/calebweinreb/multicam-calibration',
)
