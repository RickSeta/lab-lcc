from typing import List, Dict
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

TAG2SAMPLE_MAP = {
    'BB': 'Bandera Brown',
    'BG': 'Bandera Gray',
    'BN': 'Bentheimer',
    'BE': 'Berea',
    'BS': 'Berea Stripe',
    'IB': 'Boise Idaho Brown',
    'BIG': 'Boise Idaho Gray',
    'BR': 'Briarhill',
    'BU': 'Buff Berea',
    'CT': 'Carbon Tan',
    'CA': 'Castlegate',
    'CO': 'Crab Orchad',
    'IG': 'Idaho Gray',
    'IG1': 'Idaho Gray 1',
    'IG2': 'Idaho Gray 2',
    'LE': 'Leapord',
    'KI': 'Kirby',
    'KI1': 'Kirby 1',
    'KI2': 'Kirby 2',
    'NU': 'Nugget',
    'PA': 'Parker',
    'SG': 'Sister Gray Berea',
    'TO': 'Torey Buff',
    'GB': 'GB?',
    'UGB': 'UGB?',
    'AC': 'Austin Chalk',
    'DP': 'Desert Pink',
    'EW': 'Edwards White',
    'EY': 'Edwards Yellow',
    'EY1': 'Edwards Yellow 1',
    'EY2': 'Edwards Yellow 2',
    'GD': 'Guelph Dolomite',
    'IL': 'Indiana Low',
    'IM': 'Indiana Medium',
    'IH': 'Indiana High',
    'LU': 'Leuders',
    'NU': 'Nugget',
    'SD': 'Sillurian Dolomite',
    'SD2': 'Sillurian Dolomite 2',
    'WI': 'Wiscosin',
}

def parse_experimental_results(simfile: str):
    sample = simfile.split("_")[1]
    lines = []
    with open(simfile) as f:
        lines = f.readlines()
    array = []
    for line in lines:
        new_line = []
        striped = line.strip().split("   ")
        for e in striped:
            new_line.append(float(e))
        array.append(new_line)   
    df_exp = pd.DataFrame(array, columns = ["time", "diff_length", "D_D0", "uncertainty"])
    return {'sample': sample, 'data': df_exp}

def parse_sim_results(simfile: str, cols: List[str]):
    data = {}
    df = pd.read_csv(simfile)
    for col in cols:
        data[col] = df[col].values
    return data

def parse_sim_params(simfile: str, cols: List[str]):
    data = {}

    return data

def parse_dirname(dirname: str):
    data = {}
    tokens = dirname.split('/')[-1].split('_')
    data['sample'] = tokens[2]
    for t in tokens[3:]:
        k,v = t.split('=')
        data[k] = v
    return data
    

def pade_approx(times: np.ndarray, 
                D0: float, 
                svp: float, 
                tort_limit: float, 
                tort_time: float) -> np.ndarray:
    ''' 
    This function returns a Padé approximation of the diffusivity function
    '''
    term_a = (1.0 - tort_limit)
    term_b = ((4.0 * np.sqrt(D0)) / (9.0 * np.pi)) * svp
    term_sqrt = term_b * np.sqrt(times)
    term_linear = term_a * (D0/(D0*tort_time)) * times
    term_div = term_a + term_sqrt + term_linear
    return (1.0 - term_a * (term_sqrt + term_linear) / term_div)

def fit_pade_params(times: np.ndarray, 
                    ydata: np.ndarray, 
                    D0: float, 
                    bounds: List[List] = None, 
                    refac: bool = False) -> Dict:
    '''
    This functions returns Padé aproximation params for diffusivity curve
    '''
    fitting_bounds = [[0.999999*D0], [D0]]
    if(bounds):
        fitting_bounds[0].extend(bounds[0])
        fitting_bounds[1].extend(bounds[1])
    else:
        fitting_bounds[0].extend([0.0, 0.0, 0.0])
        fitting_bounds[1].extend([np.inf, 1.0, np.inf])
        
    popt, pcov = curve_fit(pade_approx, times, ydata, bounds=tuple(fitting_bounds))
    if(refac):
        r = 3.0 / popt[1]
        theta_min = r**2 / D0
        theta_max = (10.0 * r)**2 / D0
        fitting_bounds[0][3] = theta_min
        fitting_bounds[1][3] = theta_max
        popt, pcov = curve_fit(pade_approx, times, ydata, bounds=tuple(fitting_bounds))
    
    params = {
        'D0': popt[0],
        'SVp': popt[1],
        'tort_limit': popt[2],
        'theta': popt[3],
        'popt': popt,
        'pcov': pcov
    }
    return params

def order_files_by_last_token(files):
    indexes = [int(f.split('_')[-1].split('.')[0]) for f in files]
    ordered = len(files)*['']
    for i, efi in enumerate(indexes):
        ordered[efi] = files[i]
    return ordered


def fit_pfg_diffusion(Mkt, k, time, delta, threshold=0.9, cutoff=1, min_values=2):
    
    xdata = k * k
    xdata = (-1) * (time - delta/3.0) * xdata
        
    M0 = Mkt[0]
    ydata = (1.0/M0) * Mkt
    
    cutoff = find_cutoff_point(ydata, threshold, cutoff, min_values)
    
    ydata = np.log(ydata)
    Dt_fit = linear_regression_numpy(xdata[:cutoff], ydata[:cutoff], fit_intercept=False)
    return Dt_fit[0]


def find_cutoff_point(data, threshold, cutoff, min_values):
    if cutoff < 2:
        curr = 1
        while data[curr] > threshold:
            curr += 1

        cutoff = curr + 1
        if curr > data.shape[0]:
            cutoff = curr

    if cutoff < min_values:
        cutoff = min_values
    return cutoff


def linear_regression_numpy(x, y, fit_intercept=True):
    
    # Add a column of ones to x for the intercept term
    A = np.vstack([x, np.ones(len(x))]).T
    
    if fit_intercept == False:
        A[:,1] = np.zeros_like(x)

    # Use the formula beta = (X^T X)^(-1) X^T y to calculate the coefficients
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    return slope, intercept
