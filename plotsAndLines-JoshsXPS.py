import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from astropy import modeling
from scipy import stats
import statsmodels as sm

class LineData:

    xraw : np.ndarray
    yraw : np.ndarray

    xBins : np.ndarray
    yBinned : np.ndarray
    xBinned: np.ndarray
    fd_dx : np.ndarray
    model = None
    color = None
    yErrorRaw : np.ndarray
    yError : np.ndarray

    def __init__(self, x,y, cfield=None, label=None, mask=None, color=None, bins=100, calculateModel=False, yErrorRaw = None, useWeightedMean= False, showStandardError = True):
        self.xraw = x
        self.yraw = y
        self.label = label
        self.mask = mask
        self.cfield = cfield
        self.yErrorRaw = yErrorRaw
        self.useWeightedMean = useWeightedMean
        self.showStandardError = showStandardError
        self.ApplyMask()
        self.CalculateYXBinned(bins)

        if calculateModel:
            print('argument does not do anything')

    def ApplyMask(self):
        if type(self.mask) is not type(None):
            self.xraw = self.xraw[self.mask]
            self.yraw = self.yraw[self.mask]
            
            if type(self.cfield) is np.ndarray:
                self.cfield = self.cfield[self.mask]

    def CalculateYXBinned(self,bins):
        self.yBinned = np.zeros(bins)
        self.yError = np.zeros(bins)
        self.xBins = np.histogram_bin_edges(self.xraw, bins=bins)

        self.xBinned = (self.xBins[1:] + self.xBins[:-1]) / 2 #The internet says this works. We shall see

        for i in range(bins):
            mask = np.where((self.xBins[i] < self.xraw) & (self.xraw < self.xBins[i+1]), True, False)
            mask = np.logical_and.reduce((mask, np.isfinite(self.yraw)))
            if np.any(mask):
                self.yBinned[i] = np.mean(self.yraw[mask])
                self.yError[i] = np.std(self.yraw[mask], mean=self.yBinned[i])
                if self.useWeightedMean:
                    self.yBinned[i] = np.mean(self.yraw[mask])
                    self.yError[i] = np.std(self.yraw[mask], mean=self.yBinned[i])
            else:
                self.yBinned[i] = np.nan
                self.yError[i] = np.nan

        self.yBinned = np.ma.masked_invalid(self.yBinned)
        self.yError = np.ma.masked_invalid(self.yError)

        if (~self.yBinned.mask).sum() == 0: 
            print(f"{self.label} has no valid yBinned entries")

    def CalculateCorrelationStats(self):
        self.rawPearson = stats.pearsonr(self.xraw, self.yraw)
        self.sigPearson = self.rawPearson.pvalue < 0.05
        self.rawSpearman = stats.spearmanr(self.xraw, self.yraw)
        self.sigSpearman = self.rawSpearman.pvalue < 0.05


    def CalculateFiniteDerivative(self):
        self.fd_dx = np.gradient(self.yBinned, np.diff(self.xBins)[0]) #The distance between bins is uniform, so the difference should be the same

    def CalculateLinearRegression(self):
        self.model = modeling.models.Linear1D()
        self.model = modeling.fitting.LinearLSQFitter(True)(self.model, self.xBinned, self.yBinned)
        self.pearson = stats.pearsonr(self.yBinned, self.xBinned)
        self.pearsonp = np.format_float_scientific(self.pearson.pvalue, precision=4)
        self.pearsonc = round(self.pearson.statistic, 3)

    def CalculateRawLinearRegression(self):
        self.rawModel = modeling.models.Linear1D()
        self.rawModel = modeling.fitting.LinearLSQFitter(True)(self.model, self.xBinned, self.yBinned)
        self.rawPearson = stats.pearsonr(self.yBinned, self.xBinned)
        self.rawPearsonp = np.format_float_scientific(self.pearson.pvalue, precision=4)
        self.rawPearsonc = round(self.pearson.statistic, 3)

    def CalculateStatModelLinearRegression(self):
        
        pass

    def Plot(self, ax : plt.Axes, i = 0):
        stylePos = ['-','--','-.']
        if self.showStandardError:
            ax.errorbar(self.xBinned, self.yBinned, self.yError, errorevery=(i,1), fmt='none')
        ax.plot(self.xBinned, self.yBinned, stylePos[i%3], linewidth=3, label=self.label, color=self.color)

    def PlotLinearRegression(self, ax, i = 0):
        self.CalculateLinearRegression()
        stylePos = ['-','--','-.']
        ax.plot(self.xBinned, self.model(self.xBinned), stylePos[i%3], linewidth=3, label=self.label + f": m = {np.format_float_scientific(self.model.slope.value, precision=4)} \n Pearson = {self.pearsonc}, p = {self.pearsonp}", color=self.color)

    def PlotRawLinearRegression(self,ax,i = 0):
        pass

    def PlotDerivative(self,ax,i=0):
        self.CalculateFiniteDerivative()
        stylePos = ['-','--','-.']
        ax.plot(self.xBinned, self.fd_dx, stylePos[i%3], linewidth=3, label=self.label, color=self.color)

    def AddCorrelationText(self, ax: plt.Axes):
        if not (self.sigPearson and self.sigSpearman): 
            raise Exception
        
        ax.annotate(f'Spearman: {round(self.rawSpearman.statistic,3)}, Pearson: {round(self.rawPearson.statistic,3)}', xycoords='axes fraction', xy=(0.05,0.9))

    def ColorizeCField(lineDatas, mapName):
        fieldsMin, fieldsMax = np.inf, -np.inf
        for line in lineDatas:
            field= line.cfield
            fieldsMax = max((field, fieldsMax))
            fieldsMin = min((field, fieldsMin))

        norm = colors.LogNorm(vmin=fieldsMin, vmax=fieldsMax)
        cmap = mpl.colormaps['Spectral']                            #NOTE this is wrong and should probably be fixed

        for line in lineDatas:
            line.color = cmap(norm(line.cfield))

    def Sort(lineData, sortType):
        """Sort linedata according to sort type. 
        
        sortType = 'fd_dx' | 'slope' | 'spearman' | 'pearson' '"""
        if sortType == 'fd_dx':
            for line in lineData:
                line.CalculateFiniteDerivative()
            return sorted(lineData, key=lambda x : np.mean(x.fd_dx), reverse=True)
        elif sortType == 'slope':
            for line in lineData:
                line.CalculateLinearRegression()
            return sorted(lineData,  key=lambda x : np.abs(x.model.slope.value), reverse=True)
        elif sortType == 'pearson':
            for line in lineData:
                line.CalculateCorrelationStats()
            return sorted(lineData, key= lambda x: np.abs(x.rawPearson.statistic), reverse=True)
        elif sortType == 'spearman':
            for line in lineData:
                line.CalculateCorrelationStats()
            return sorted(lineData, key= lambda x: np.abs(x.rawSpearman.statistic), reverse=True)
        

class PlotContainer:
    titleAndUnitObj : object
    axesPlotterObj : object
    axesControllerObj : object
    def __init__(self, axesPlotter, titleAndUnit, axesController, figSize = (8.5,11)):
        self.axesControllerObj = axesController
        self.axesPlotterObj = axesPlotter
        self.titleAndUnitObj = titleAndUnit
        self.figSize = figSize

    def __call__(self, lineDataTuple : ()):
        self.axesControllerObj.ConfigureAxesAndPlot(self.axesPlotterObj, self.titleAndUnitObj, lineDataTuple, self.figSize)
        plt.show()

    class TitleAndUnitLabels:
        def __init__(self,title, xLabel,yLabel, yRange = None):
            self.title = title
            self.xLabel = xLabel
            self.yLabel = yLabel
            self.yRange = yRange
        
        def SetAxes(self, ax : plt.Axes):
            ax.set_xlabel(xlabel=self.xLabel)
            ax.set_ylabel(self.yLabel)

            ax.set_ylim(self.yRange)

        def SetFigTitle(self, fig : plt.Figure):
            fig.suptitle(self.title)

    class AxesPlotter:
        def __init__(self, plottingFunctions : tuple, showCorrelations = False):
            self.plottingFunctions = plottingFunctions
            self.showCorrelations = showCorrelations
        def Plot(self, lineData : list[LineData], ax):
            i = 0
            for line in lineData:
                for func in self.plottingFunctions:
                    func(line, ax, i)
                    if self.showCorrelations:
                        line.AddCorrelationText(ax)
                i += 1
        
        def StdPlotFunc(line, ax, i):
            line.Plot(ax, i=i)
        def DerivPlotFunc(line : LineData, ax, i):
            line.PlotDerivative(ax,i=i)
        def LinearRegressPlotFunc(line :LineData, ax,i):
            line.PlotLinearRegression(ax,i=i)
            
    class AxesController:
        
        def __init__(self, ncols, nrows, numPerAxes):
            self.ncols = ncols
            self.nrows = nrows
            self.numPerAxes = numPerAxes

        def ConfigureAxesAndPlot(self, AxesPlotter, TitleandUnit, lineDataTuple : tuple[list[LineData]], figSize = (8.5,11)):
            fig, axArray = plt.subplots(figsize=figSize, ncols=self.ncols, nrows=self.nrows, sharex=True, sharey=True)
            for lineData in lineDataTuple:
                i = 0
                for ax in axArray.flat:
                    AxesPlotter.Plot(lineData[i:i+self.numPerAxes], ax)
                    i += self.numPerAxes
                    TitleandUnit.SetAxes(ax)
                    ax.legend()

            TitleandUnit.SetFigTitle(fig)

            fig.tight_layout()

            return fig, axArray
   

            
                    