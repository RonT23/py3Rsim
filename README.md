# 3R Spatial Robot Manipulator Simulator

## Folder Structure

```text
        py3Rsim/
        
        \__ py3Rsim_package/
                
                \__ py3Rsim_package/
                        \__ __init__.py
                        \__ robot_sim.py
                
                \__ README.md
                \__ setup.py

        \__ results/
                
        \__ README.md
        \__ main.py
        \__ user.py
        \__ docs/
                \__ trajectory_planning_and_sim.pdf
```

## Description

TO DO!

## Prerequisites

This package is written in Python, making it largely hardware- and operating system-agnostic, so it should operate similarly across different systems. However, as development and testing were conducted in a Linux-based environment, it is recommend using a Linux environment to run the simulator. Most commands should also work on Windows via Windows Subsystem for Linux (WSL), or by using a virtual machine on a Windows host running one of the many Linux distros available.

The only requirement is Python version 3.5 or higher. Other than that, there are no additional dependencies to run the simulator. The required packages are installed if they are not already istalled through the package itself.

## Create Virtual Environment

To keep this package separate from other Python packages on your system, it is best practice to use a virtual environment. To create and activate a virtual environment run:

```shell
    $ python3 -m venv py3Rsim_env
    $ source ./py3Rsim_env/bin/activate
```

## Package Installation

Navigate to the `./py3Rsim_package` directory and run the command to install the package locally on your computer (within the virtual environment you just created).

```shell
    $ pip install -e .
```

## Run Simulation

## Simulation Output

## Change Task Parameters

## Change Internal Parameters

## Documentation
More details can be found in `trajectory_planning_and_sim.pdf` located in the `./docs/` folder. Please note that the document is written in Greek! For further information please contact me at `rontsela@mail.ntua.gr` or `ron-tsela@di.uoa.gr`. 