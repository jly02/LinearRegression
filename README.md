# regpy
A small library for creating and plotting regression models.

regpy aims to make it as painless as possible to create and plot regression models, making it a great tool for incredibly rapid development over small datasets or as an educational tool for demonstrations.

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


To create a linear regression, initialize a LinearRegression object, which requires two lists. The only restriction on these lists is that they cannot be empty.
```py
import regpy.linreg as lrp
# This should create a linear regression that can be represented as y = x on a graph
linReg = lrp.LinearRegression([1, 2, 3], [1, 2, 3])
```


To make a prediction based on the regression line, use the predict() method, which is a member of the LinearRegression class.
```py
import regpy.linreg as lrp
linReg = lrp.LinearRegression([1, 2, 3], [1, 2, 3])
print(linReg.predict(-1)) # Will print -1.0
print(linReg.predict(0)) # Will print 0.0
print(linReg.predict(10000)) # Will print 10000.0
```


You can also access the R-Squared value for the regression using the public fields on a LinearRegression object.
```py
import regpy.linreg as lrp
linReg = lrp.LinearRegression([1, 2, 3], [1, 2, 3])
print(f"R^2 = {linReg.rSquare}") # R^2 = 1.0
```


Finally, printing a LinearRegression will give you the line equation in the form of y = {beta}x + {inty}, both rounded to two decimal places (beta and inty, as well as the domain and range of the dataset, are also freely accessible).
```py
import regpy.linreg as lrp
linReg = lrp.LinearRegression([1, 2, 3], [1, 2, 3])
print(linReg) # y = 1.00x + 0.00
```

Plotting
--------
Plotting a regression line is done internally using matplotlib, and only requires two functions to be called.
```py
import regpy.linreg as lrp
linReg = lrp.LinearRegression([1, 2, 3], [1, 2, 3])
linReg2 = lrp.LinearRegression([2, 3, 4], [2, 3, 4])
lrp.plotLinReg(linReg) # queues the first regression line to be plotted
lrp.plotLinReg(linReg2) # queues the second regression line to be plotted
lrp.drawPlots() # displays the plot
```

Plotting the scattered points of the dataset alongside the line is optional, and so is setting a label for the x and y axes. By default, the scatter is always plotted, and the x and y axes are labeled "x-values" and "y-values" respectively. The title of each plot will always be the string representation of the given LinearRegression. 
```py
import regpy.linreg as lrp
linReg = lrp.LinearRegression([1, 2, 3], [1, 2, 3])
lrp.plotLinReg(linReg, xLabel="x-axis", yLabel="y-axis", scatter=False) # optional parameters
lrp.drawPlots()
```

Notes
-----
Some may have still have issues importing this package after installation, to which I would recommend using a virtual environment:

Install venv using `pip install venv` or `pip3 install venv`

To create a virtual environment, run `python3 -m venv venv`

Activate it with `venv/Scripts/activate` (for Windows) or `./venv/Scripts/activate` (for Linux-based)

Run `pip install -e .` again in the root directory and it should work then. To close the virtual environment just use `deactivate`

>regpy provides a set of tests in the `test/` directory to ensure that it doesn't just run on my machine. If all tests pass you can be reasonably confident it will work for your use cases

>This library is written purely in python, so expect performance bottlenecks should you decide to apply this library to large-scale data sets. It is also a WIP, so there may be new types of regressions that are added to the library in the future.
