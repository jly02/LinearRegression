""" The linregpy module provides a simple way to creating
linear regression models, with a wrapper API to matplotlib
for plotting the regressions and their scatter-plots 
painlessly.
"""


from math import ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from dataclasses import dataclass


# global variables
_plots = [] # stores the plots that have been requested to be drawn
        
        
class LinearRegression:
    """ A class representing a linear regression over a set of given
    2-dimensional data.
    
    Attributes
    ----------
    beta : float
        the slope coefficient of the regression line
    inty : float
        the y-intercept of the regression line
    domain : (float, float)
        the min/max x-values in the linear regression
    range : (float, float)
        the min/max y-values in the linear regression
    """
    
    def __init__(self, x: List[float], y: List[float]) -> None:
        if len(x) == 0 or len(y) == 0:
            raise ValueError("axes cannot be empty")
            
        self._xAxis = x
        self._yAxis = y
        self.beta   = self._calculateBeta()
        self.inty   = self._calculateIntercept()
        self.domain = (min(x), max(x))
        self.range  = (min(y), max(y))
        self.rSquare = self._calcRSquare()
        self.rSquareAdj = self._calcRSquareAdj()
     
       
    def _calculateBeta(self) -> float:
        """ Calculates the slope coefficient.

        Returns
        -------
        float
            the slope coefficient 
        """
        xBar = sum(self._xAxis) / len(self._xAxis)
        xErr = []
        for x in self._xAxis:
            xErr.append(x - xBar)
            
        yBar = sum(self._yAxis) / len(self._yAxis)
        yErr = []
        for y in self._yAxis:
            yErr.append(y - yBar)
            
        # just need the result of the dot product for covariance of x and y
        covXY = np.dot(xErr, yErr)
        # variance of x
        varX = sum(x**2 for x in xErr)
        
        return covXY / varX
    
    
    def _calculateIntercept(self) -> float:
        """ Calculates the y-intercept.

        Returns
        -------
        float
            the y-intercept
        """
        xBar = sum(self._xAxis) / len(self._xAxis)
        yBar = sum(self._yAxis) / len(self._yAxis)
        
        return yBar - (self.beta * xBar)
    
    
    def _calcRSquare(self) -> float:
        """ Calculates the R-squared (coefficient of determination)
        of this model over the given dataset.

        Returns:
            float: the r-squared value
        """
        yPred = [self.predict(x) for x in self._xAxis]
        yBar = sum(self._yAxis) / len(self._yAxis)
        # sum squared of prediction error and total sum
        ssReg = []
        ssTotal = []
        for y, yHat in zip(self._yAxis, yPred):
            ssReg.append((y - yHat)**2)
            ssTotal.append((y - yBar)**2)
            
        return 1 - (sum(ssReg) / sum(ssTotal))
    
    
    def _calcRSquareAdj(self) -> float:
        """ Calculates the adjusted R-squared of this model.

        Returns:
            float: the adjusted r-squared value
        """
        # The assumption of a single independent variable is made.
        sampleSize = len(self._xAxis)
        return 1 - (((1 - self.rSquare) * (sampleSize - 1)) / (sampleSize - 2))
    
    
    def predict(self, x: float) -> float:
        """ Returns a prediction given by the regression line based on a given
        value.

        Arguments
        ---------
        x : float
            the value of the independent variable to make a prediction at

        Returns
        -------
        float: 
            the value of the regression line evaluated at the given point
        """
        return (self.beta * x) + self.inty
    
    
    def __repr__(self) -> str:
        return "y = {:0.2f}x + {:0.2f}".format(self.beta, self.inty)
    
    
@dataclass(frozen=True, order=True)
class _LinRegPlot:
    # Wrapper class for storing linear regressions before they are plotted
    linReg: LinearRegression
    xLabel: str
    yLabel: str
    scatter: bool
    

def plotLinReg(linReg: LinearRegression, xLabel="x-values", yLabel="y-values", scatter=True) -> None:
    """ Enqueues a linear regression to be plotted.

    Arguments
    ---------
    linReg : LinearRegression
        the linear regression to be plotted
    xLabel : str, optional
        the x-axis label. Defaults to "x-values".
    yLabel : str, optional
        the y-axis label. Defaults to "y-values".
    """
    global _plots
    _plots.append(_LinRegPlot(linReg, xLabel, yLabel, scatter))


def drawPlots() -> None:
    """ Shows all of the queued plots.
    """
    global _plots
    # Calculating optimal number of rows and columns given N (numPlots) cells
    numPlots = len(_plots)
    cols = int(sqrt(numPlots))
    rows = int(ceil(numPlots / cols))
    fig, axes = plt.subplots(cols, rows)
    currPlot = 0
    
    # a single plot can just be plotted on its own without any special considerations
    if(numPlots == 1):
        linReg = _plots[currPlot].linReg
        var = np.linspace(linReg.domain[0], linReg.domain[1], 10) # since this is linear, you can choose any number of points
        func = linReg.beta * var + linReg.inty
        plt.plot(var, func)
        
        # plotting the scatter is optional
        if(_plots[currPlot].scatter == True):
            plt.scatter(linReg._xAxis, linReg._yAxis)
            
        plt.xlabel(_plots[0].xLabel)
        plt.ylabel(_plots[0].yLabel)
        plt.title(f"Plot of {linReg}")
        
    # two or three plots are simply calculated as a 1-D array
    elif(numPlots < 4):
        for row in range(rows):
            linReg = _plots[currPlot].linReg
            var = np.linspace(linReg.domain[0], linReg.domain[1], 10) # since this is linear, you can choose any number of points
            func = linReg.beta * var + linReg.inty
            axes[row].plot(var, func)
            
            # plotting the scatter is optional
            if(_plots[currPlot].scatter == True):
                axes[row].scatter(linReg._xAxis, linReg._yAxis)
                
            axes[row].set_xlabel(_plots[0].xLabel)
            axes[row].set_ylabel(_plots[0].yLabel)
            axes[row].set_title(f"Plot of {linReg}")
            currPlot += 1
            
    # any more than three plots become a 2-D array
    else:
        for row in range(rows):
            for col in range(cols):
                # If the plot doesn't exist we can just delete excess axes
                try:
                    linReg = _plots[currPlot].linReg
                except:
                    fig.delaxes(axes[col][row])
                    continue
                    
                var = np.linspace(linReg.domain[0], linReg.domain[1], 10) # since this is linear, you can choose any number of points
                func = linReg.beta * var + linReg.inty
                axes[col][row].plot(var, func)
                    
                # plotting the scatter is optional
                if(_plots[currPlot].scatter == True):
                    axes[col][row].scatter(linReg._xAxis, linReg._yAxis)
                    
                axes[col][row].set_xlabel(_plots[0].xLabel)
                axes[col][row].set_ylabel(_plots[0].yLabel)
                axes[col][row].set_title(f"Plot of {linReg}")
                currPlot += 1
                
    fig.suptitle("Note: Scales are not the same.")
    plt.tight_layout()
    plt.show()
    
    
# Nothing to see here. For now.
def __main():
    print("Hello, linregpy!")
    

# Main is not run if this file is imported as a library.
if __name__ == "__main__":
    __main()