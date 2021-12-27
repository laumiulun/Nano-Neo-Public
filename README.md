# Nano_Neo
#### Versions: 0.0.4
#### Last update: Sep 14, 2021

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

TBA
