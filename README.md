# regpy
A small library for creating and plotting regression models.

regpy aims to make it as painless as possible to create and plot regression models, making it a great tool for incredibly rapid development over small datasets or as an educational tool for demonstration.

Setup
-----
Run `pip install -e .` in the root directory and you should be good to go!

Getting Started
---------------
To begin using regpy, import the package:
```py
import regpy.linreg as lrp
```
To ensure that the the module as been correctly imported, run it. If you get a ModuleNotFoundError, refer to the Notes section of this README.

To create a linear regression, initialize a LinearRegression object, which takes in two lists as parameters. The only restriction on these lists is that they cannot be empty.
```py
# This should create a linear regression that can be represented as y = x on a graph
linReg = lrp.LinearRegression([1, 2, 3], [1, 2, 3])
```

Notes
-----
Some may have still have issues importing this package after installation, to which I would recommend using a virtual environment:

Install venv using `pip install venv` or `pip3 install venv`

To create a virtual environment, run `python3 -m venv venv`

Activate it with `venv/Scripts/activate` (for Windows) or `./venv/Scripts/activate` (for Linux-based)

Run `pip install -e .` again in the root directory and it should work then. To close the virtual environment just use `deactivate`

>This library is written purely in python, so expect performance bottlenecks should you decide to apply this library to large-scale data sets
>It is also a WIP, so there may be new types of regressions that are added to the library in the future.