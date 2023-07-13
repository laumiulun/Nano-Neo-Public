# Nano_Neo
#### Versions: 0.0.5
#### Last update: Jul 12, 2023

This program utilize Genetic algorithm in fitting of Nano_Indent

## Pre-requisites
Usage of this software is highly recommend to use `anaconda` or `pip` package managers.

  - Python: 3.7>
  - Numpy: 1.17.2>
  - Matplotlib: 3.1.2>

It is highly recommend to create a new environment in `anaconda` to prevent packages conflicts.

        conda create --name nano_neo python=3.7 numpy matplotlib pyqt psutil
        conda activate nano_neo


## Installations
To install Nano_Neo, simply clone the repo:

        git clone https://github.com/laumiulun/nano-indent.git
        cd nano-indent
        python setup.py install


## Usage
To run a sample test, make sure the environment is set correctly, and select a input file:

        nano_neo -i test/test.ini

## Update
Nano Neo is under active development, to update the code after pulling from the repository:

        git pull
        python setup.py install

## GUI

We also have provided a GUI for use in additions to our program, with additional helper script to facilitate post-analysis. To use the GUI:

        cd gui
        python nano_neo_gui.py

The GUI allows for custom parameters for different indentator during the post-analysis process.

## Potential Errors
If you get an error message involving psutl, make sure you are in the right conda environment and install psutil

        conda activate nano_neo
        conda install psutl

## Citation:

[References](https://www.sciencedirect.com/science/article/pii/S0169433222032627)

Harvard:

        Burleigh, A., Lau, M.L., Burrill, M., Olive, D.T., Gigax, J.G., Li, N., Saleh, T.A., Pellemoine, F., Bidhar, S., Long, M. and Ammigan, K., 2023. Artificial intelligence based analysis of nanoindentation load–displacement data using a genetic algorithm. Applied Surface Science, 612, p.155734.

BibTex:

        @article{burleigh2023artificial,
        title={Artificial intelligence based analysis of nanoindentation load--displacement data using a genetic algorithm},
        author={Burleigh, Abraham and Lau, Miu Lun and Burrill, Megan and Olive, Daniel T and Gigax, Jonathan G and Li, Nan and Saleh, Tarik A and Pellemoine, Frederique and Bidhar, Sujit and Long, Min and others},
        journal={Applied Surface Science},
        volume={612},
        pages={155734},
        year={2023},
        publisher={Elsevier}
        }
