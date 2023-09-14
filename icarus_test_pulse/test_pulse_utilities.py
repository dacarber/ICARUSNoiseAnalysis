import numpy as np
import matplotlib.pyplot as plt
import uproot
from scipy.special import jv
from scipy.optimize import curve_fit
import scipy.integrate as integrate
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.stats import chisquare
#from mpmath import nsum


def load_waveform_file(path, run, fragment, evt) -> np.array:
    """
    Loads waveforms from a file. Input files should be comma separated
    with each waveform in the crate having its own separate line in
    the file.

    Parameters
    ----------
    path: str
        The base path to the input files.
    run: int
        The run number.
    fragment: int
        The integer representation of the fragment ID.
    evt: int
        The event number.

    Returns
    -------
    waveforms: numpy.array
        The waveforms for the component with shape (576,4096) (nominal)
        or (512, 4096) (top crate).
    """
    with open(f'{path}run{run}_frag{fragment}_evt{evt}') as input_file:
        lines = input_file.readlines()
    waveforms = np.array([x.strip('\n').split(',') for x in lines], dtype=float)
    return waveforms

def align_and_average(waveforms, max=True, size=1250) -> np.array:
    """
    Calculate the average pulse. First, split the waveforms into
    separate, individual pulses. Afterwards, alignment using the min
    or max of the each piece is performed by "rolling" the waveform.
    The final step is to calculate the mean of each piece.

    Parameters
    ----------
    waveforms: numpy.array
        The input waveforms with shape (N, 4096).
    max: bool
        Boolean flag for using max(min) for alignment.
    size: int
        The number of ticks comprising a single pulse.

    Returns
    -------
    waveform: numpy.array
        The resulting average waveform.
    """
    waveforms = np.vstack(np.split(waveforms, [size*i for i in range(1, int(4096/size)+1)]))
    waveform = np.mean(np.roll(waveforms, np.argmax(waveforms, axis=1), axis=0), axis=0)
    tcenter = np.argmax(waveform) if max else np.argmin(waveform)
    waveform = np.roll(waveform, 75-tcenter)[0:150]
    return waveform

def average_pulse(path, run, fragment, ch, title, evt=1, scale=2200, internal=False, nchannels=64) -> None:
    """
    Plots the average test pulse shape for both the positive lobe and
    the inverted negative lobe. Waveform input files should be comma
    separated with each waveform in a crate having its own separate
    line in the file.

    Parameters
    ----------
    path: str
        The base path to the input files.  
    run: int
        The run number of the test pulse input.
    fragment: int
        The integer representation of the fragment ID.
    ch: int
        The first channel in the pulsed board. If external, the entire
        board is used. If internal, only the channels on the board
        congruent to the channel mod 2 are used.
    title: str
        The title of the plot
    evt: int
        The number of the event.
    scale: float
        The y-range of the plot. The range will be set to (-250, scale).
    internal: bool
        Boolean flag denoting internal vs. external test pulse.
    nchannels: int
        Number of channels to range over for the averaging. 

    Returns
    -------
    None.
    """
    plt.style.use('../plot_style.mplstyle')

    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()

    waveforms = load_waveform_file(path, run, fragment, evt)[ch:ch+nchannels:2 if internal else 1,:]
    pwaveform = align_and_average(waveforms, max=True, size=1250)
    mwaveform = -1 * align_and_average(waveforms, max=False, size=1250)

    ax.plot(np.arange(150), pwaveform, linestyle='-', linewidth=2, label='Positive Lobe')
    ax.plot(np.arange(150), mwaveform, linestyle='-', linewidth=2, label='Negative Lobe')
    ax.set_xlim(0,150)
    ax.set_ylim(-250, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    ax.legend()
    figure.suptitle(title)

def average_waveform(path, run, fragment, ch, title, evt=1, scale=2200, internal=False, nchannels=64):
    """
    Plots the average test pulse waveform. Waveform input files should be comma
    separated with each waveform in a crate having its own separate
    line in the file.

    Parameters
    ----------
    path: str
        The base path to the input files.  
    run: int
        The run number of the test pulse input.
    fragment: int
        The integer representation of the fragment ID.
    ch: int
        The first channel in the pulsed board. If external, the entire
        board is used. If internal, only the channels on the board
        congruent to the channel mod 2 are used.
    title: str
        The title of the plot
    evt: int
        The number of the event.
    scale: float
        The y-range of the plot. The range will be set to (-scale, scale).
    internal: bool
        Boolean flag denoting internal vs. external test pulse.
    nchannels: int
        Number of channels to range over for the averaging. 

    Returns
    -------
    None.
    """
    plt.style.use('../plot_style.mplstyle')

    figure = plt.figure(figsize=(14,6))
    ax = figure.add_subplot()

    waveforms = load_waveform_file(path, run, fragment, evt)[ch:ch+nchannels:2 if internal else 1,:]
    waveform = np.mean(np.roll(waveforms, np.argmax(waveforms, axis=1), axis=0), axis=0)
    
    ax.plot(np.arange(4096), waveform, linestyle='-', linewidth=1)
    ax.set_xlim(0,4096)
    ax.set_ylim(-scale, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    figure.suptitle(title)

def compare_average_pulse(path, runs, fragments, chs, strides, nchannels, labels, title, evt=1, scale=2200) -> None:
    """
    Plots the average test pulse shapes for each of the input runs.
    Waveform input files should be comma separated with each waveform
    a crate having its own separate line in the file.

    Parameters
    ----------
    path: str
        The base path to the input files.
    runs: list[int]
        The list of run numbers for the test pulse input.
    fragments: list[int]
        The list of the fragment IDs of the pulsed crates.
    chs: list[int]
        The list of the first channels in each of the pulsed crates.
    strides: list[int]
        The list of strides to use when selecting channels from the
        input waveforms (e.g. internal test pulses use every other
        channel, so stride=2).
    nchannels: list[int]
        The list of the number of channels to range over for the
        averaging.
    labels: list[str]
        The list of labels to use in the legend.
    title: str
        The title of the plot.
    evt: int
        The number of the event.
    scale: float
        The y-range of the plot. The range will be set to (-250, scale).

    Returns
    -------
    None.
    """

    plt.style.use('../plot_style.mplstyle')

    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()

    for ri, r in enumerate(runs):
        waveforms = load_waveform_file(path, r, fragments[ri], evt)[chs[ri]:chs[ri]+nchannels[ri]:strides[ri],:]
        waveform = align_and_average(waveforms, max=True, size=1250)
        ax.plot(np.arange(150), waveform, linestyle='-', linewidth=2, label=labels[ri])

    ax.set_xlim(0,150)
    ax.set_ylim(-250, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    ax.legend()
    figure.suptitle(title)

def plot_single_test_waveform(path, run, fragment, title, evt, ch, scale=2200) -> None:
    """
    Plots a single waveform. Waveform input files should be comma
    separated with each waveform in a crate having its own separate
    line in the file.

    Parameters
    ----------
    path: str
        The base path to the input files.
    run: int
        The run number.
    fragment: int
        The integer representation of the fragment ID.
    title: str
        The title to place on the plot.
    scale: float
        The y-range of the plot. The range will be set to (-scale, scale).
    """
    plt.style.use('../plot_style.mplstyle')

    figure = plt.figure(figsize=(16,6))
    ax = figure.add_subplot()

    waveform = load_waveform_file(path, run, fragment, evt)[ch, :]
    ax.plot(np.arange(4096), waveform, linestyle='-', linewidth=2)
    ax.set_xlim(0,4096)
    ax.set_ylim(-scale, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    figure.suptitle(title)

def plot_average_waveform_artdaq(path, run, title, channel=0, scale=2200):
    """
    Plots the average waveform for the specified channel using a ROOT
    file prepared by the TPCTestPulseArtDAQ module. This module aligns
    waveforms and stores them (as a sum) in a TH2I with a set of bins
    for each channel.

    Parameters
    ----------
    path: str
        The base path to the input ROOT file.
    run: int
        The run number of the input ROOT file.
    title: str
        The title to place on the plot.
    channel: int
        The channel number to retrieve and plot.
    scale: float
        The y-range of the plot. The range will be set to (-scale, scale).
    """
    plt.style.use('../plot_style.mplstyle')
    
    figure = plt.figure(figsize=(16,6))
    ax = figure.add_subplot()
    
    data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
    waveform, count = data[:-1, channel].astype(float), data[-1,channel]
    waveform -= np.median(waveform)
    waveform /= count
    ax.plot(np.arange(len(waveform)), waveform, linestyle='-', linewidth=2)
    ax.set_xlim(0, len(waveform))
    ax.set_ylim(-scale, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    figure.suptitle(title)

def plot_average_pulse_artdaq(path, run, title, channel=0, scale=2200):
    """
    Plots the average waveform for the specified channel using a ROOT
    file prepared by the TPCTestPulseArtDAQ module. This module aligns
    waveforms and stores them (as a sum) in a TH2I with a set of bins
    for each channel.

    Parameters
    ----------
    path: str
        The base path to the input ROOT file.
    run: int
        The run number of the input ROOT file.
    title: str
        The title to place on the plot.
    channel: int
        The channel number to retrieve and plot.
    scale: float
        The y-range of the plot. The range will be set to (-250, scale).
    """
    plt.style.use('../plot_style.mplstyle')
    
    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()
    
    data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
    waveform, count = data[:-1, channel].astype(float), data[-1,channel]
    waveform -= np.median(waveform)
    waveform /= count

    pwaveform = waveform[:150]
    mpeak = np.argmin(waveform)
    mwaveform = -1 * waveform[mpeak-75:mpeak+75]
    ax.plot(np.arange(150), pwaveform, linestyle='-', linewidth=2, label='Positive Lobe')
    ax.plot(np.arange(150), mwaveform, linestyle='-', linewidth=2, label='Negative Lobe')
    ax.set_xlim(0, 150)
    ax.set_ylim(-250, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    ax.legend()
    figure.suptitle(title)
def bessel(x,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o):
    
    bessel_func = (jv(0,np.subtract(x,e))*a+jv(1,np.subtract(x,f))*b+jv(2,np.subtract(x,h))*c+jv(3,np.subtract(x,i))*d+jv(4,np.subtract(x,j))*k+jv(5,np.subtract(x,l))*m+jv(6,np.subtract(x,n))*o)*g
    return bessel_func
#+jv(5,np.subtract(x,l))*m+jv(6,np.subtract(x,n))*o
#def bessel_sum(x,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o):    
#    bessel_sum_func = nsum(lambda k: 
#    return bessel_func
def milind(x,phi,psi,G,shift):
#def milind(x,scaling,shift):
    '''
    p = 5.89618
    alpha = 6.65666
    beta = 2.38722
    b = 5.18561
    a = 4.80637
    A = 4.31054
    B = -5.24039
    C = -13.0014
    D = 0.929848
    E = .535356
    '''
    '''
    p = 2.94809
    alpha = 2.82833
    beta = 1.19361
    b = 2.5928
    a = 2.40318
    A = 4.31054
    B = -5.24039
    C = -13.0014
    D = 0.929848
    E = .535356
    '''
    '''
    p = 1.96539
    alpha = 1.88555
    beta = 0.795739
    b = 1.72854
    a = 1.60212
    A = 4.31054
    B = -5.24039
    C = -13.0014
    D = 0.929848
    E = .535356
    '''
    '''
    p = 1.47405
    alpha = 1.41417
    beta = 0.596804
    b = 1.2964
    a = 1.20159
    A = 4.31054
    B = -5.24039
    C = -13.0014
    D = 0.929848
    E = .535356
    '''
    
    tau = 1.5*shift
    #phi = .4#.3991
    #psi = .95#.8233
    alpha = np.cos(phi)/tau
    beta = np.sin(phi)/tau
    a = np.cos(psi)/tau
    b = np.sin(psi)/tau
    q = (((-1/tau)+a)**2+b**2)  
    Q = (((-1/tau)+alpha)**2+beta**2)
    p =1/tau
    #shift = 1
    #print("alpha: ",alpha," beta: ", beta," a: ", a," b: ", b, " q: ", q," Q: ",Q," p: ", p, "tau: ", tau, " G: ", G,"shift:" ,shift," phi: ", phi, " psi: ", psi)
    A = G/(q*Q)
    B  = (G*(-a**2-b**2-2*a*p+4*a*alpha+2*p*alpha-3*alpha**2+beta**2))/(Q*((b**2 + (a-alpha)**2)**2 + 2*(a - b -alpha)*(a + b - alpha)*beta**2+beta**4))
    C = (G*(a**2*(1/tau) + b**2 * (1/tau) - 2*a**2*alpha - 2*b**2*alpha - 4*a*(1/tau)*alpha + 6*a*alpha**2 + 3*p*alpha**2 - 4*alpha**3 - 2*a*beta**2 -(1/tau)*beta**2+4*alpha*beta**2))/(Q*((b**2+(a-alpha)**2)**2+2*(a-b-alpha)*(a+b-alpha)*beta**2+beta**4))
    D = -G*(3*a**2 - b**2 - 2*a*(1/tau) - 4*a*alpha + 2*(1/tau)*alpha + alpha**2 + beta**2)/(q*((b**2+(a-alpha)**2)**2 + 2*(a-b-alpha)*(a+b-alpha)*beta**2+beta**4))
    E = -G*(4*a**3 - 4*a*b**2 - 3*a**2*(1/tau)+b**2*(1/tau)-6*a**2*alpha + 2*b**2*alpha+4*a*(1/tau)*alpha + 2*a*alpha**2 - p*alpha**2+2*a*beta**2-(1/tau)*beta**2)/(q*((b**2+(a-alpha)**2)**2+2*(a-b-alpha)*(a+b-alpha)*beta**2+beta**4))
    
    #p = p*shift
    
    #milind_func
    #print(x)
    milind_func = (A*np.exp(-p*(x))+B*np.exp(-alpha*(x))*np.cos(beta*(x))+(C-B*alpha)*(1/beta)*np.exp(-alpha*(x))*np.sin(beta*(x))+D*np.exp(-alpha*(x))*np.cos(b*(x))+(E-D*a)*(1/b)*np.exp(-alpha*(x))*np.sin(b*(x)))#*scaling
    return milind_func
def zero_pole(x,a,c,shift,scale):
    b= 1.5*scale
    x=x-+shift
    zero_pole_func = (a/c)*((x)/(b))*np.exp(-(x)/(b))
    return zero_pole_func
def fit_bessel(path, run, title, channel=0, scale=2200):
    """
    Plots the average waveform for the specified channel using a ROOT
    file prepared by the TPCTestPulseArtDAQ module. This module aligns
    waveforms and stores them (as a sum) in a TH2I with a set of bins
    for each channel.

    Parameters
    ----------
    path: str
        The base path to the input ROOT file.
    run: int
        The run number of the input ROOT file.
    title: str
        The title to place on the plot.
    channel: int
        The channel number to retrieve and plot.
    scale: float
        The y-range of the plot. The range will be set to (-250, scale).
    """
    plt.style.use('../plot_style.mplstyle')
    
    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()
    
    data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
    waveform, count = data[:-1, channel].astype(float), data[-1,channel]
    waveform -= np.median(waveform)
    waveform /= count

    pwaveform = waveform[:150]
    mpeak = np.argmin(waveform)
    mwaveform = -1 * waveform[mpeak-75:mpeak+75]
    peak = np.argmax(pwaveform)
    
    popt, pcov = curve_fit(bessel, np.arange(47),pwaveform[peak-7:peak+40],maxfev =50000)
    #print(pwaveform)
    print(popt)
    
    fit = bessel(np.arange(0,67,.1),*popt)
    print("Peak value", fit[np.argmax(fit)],pwaveform[peak])
    #options={'limit':100}
    
    result = integrate.quad(lambda x: bessel(x,*popt), -100, 2000,limit = 2000)
    print("Integral of fit from -100 to 2000 ", result) 
    
    peaks, _ = find_peaks(fit)
    results_half = peak_widths(fit, peaks, rel_height=0.5)
    print("Peaks",peaks)
    print("Widths of peak in fit ",results_half[0])
    
    chi_square = np.sum(np.divide((np.subtract(pwaveform[peak-6:peak+40], bessel(np.arange(46),*popt))**2),bessel(np.arange(46),*popt)))
    print("Chi^2: ", chi_square)
    
    
    ax.plot(np.arange(0,67,.01), bessel(np.arange(0,67,.01),*popt), linestyle = '--', linewidth=5, label='Bessel fit')
    ax.plot(np.arange(67), pwaveform[peak-7:peak+60], linestyle='-', linewidth=2, label='Positive Lobe')
    #ax.plot(np.arange(150), mwaveform, linestyle='-', linewidth=2, label='Negative Lobe')
    ax.set_xlim(-10, 80)
    ax.set_ylim(-250, scale)
    #ax.set_ylim(-20, 200)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    ax.legend()
    figure.suptitle(title)
def fit_zero_pole(path, run, title, channel_1=0, scale=2200,cross_talk = False):
    """
    Plots the average waveform for the specified channel using a ROOT
    file prepared by the TPCTestPulseArtDAQ module. This module aligns
    waveforms and stores them (as a sum) in a TH2I with a set of bins
    for each channel.

    Parameters
    ----------
    path: str
        The base path to the input ROOT file.
    run: int
        The run number of the input ROOT file.
    title: str
        The title to place on the plot.
    channel: int
        The channel number to retrieve and plot.
    scale: float
        The y-range of the plot. The range will be set to (-250, scale).
    """
    plt.style.use('../plot_style.mplstyle')
    
    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()
    
    if cross_talk == True:
        channel_2 = int(input("Enter the other channel:"))
        plt.style.use('../plot_style.mplstyle')

        data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
        waveform_1, count_1 = data[:-1, channel_1].astype(float), data[-1,channel_1]
        waveform_1 -= np.median(waveform_1)
        waveform_1 /= count_1
        waveform_2, count_2 = data[:-1, channel_2].astype(float), data[-1,channel_2]
        waveform_2 -= np.median(waveform_2)
        waveform_2 /= count_2
        waveform = np.subtract(waveform_1,waveform_2)
    else:
        data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
        waveform, count = data[:-1, channel_1].astype(float), data[-1,channel_1]
        waveform -= np.median(waveform)
        waveform /= count

    pwaveform = waveform[:150]
    mpeak = np.argmin(waveform)
    mwaveform = -1 * waveform[mpeak-75:mpeak+75]
    peak = np.argmax(pwaveform)
    #start = 
    popt, pcov = curve_fit(zero_pole, np.arange(44),pwaveform[peak-4:peak+40],maxfev =5000)
    print(pwaveform)
    print(popt)
    ax.plot(np.arange(67), zero_pole(np.arange(67),*popt), linestyle = '--', linewidth=5, label='ICARUS Electronics fit')
    ax.plot(np.arange(67), pwaveform[peak-7:peak+60], linestyle='-', linewidth=2, label='Positive Lobe')
    #ax.plot(np.arange(150), mwaveform, linestyle='-', linewidth=2, label='Negative Lobe')
    ax.set_xlim(-10, 80)
    ax.set_ylim(-250, scale)
    #ax.set_ylim(-10, 200)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    ax.legend()
    figure.suptitle(title)
def list_bessel_fit(path, run, title, channel_start=0,channel_end = 1, scale=2200):
    """
    Plots the average waveform for the specified channel using a ROOT
    file prepared by the TPCTestPulseArtDAQ module. This module aligns
    waveforms and stores them (as a sum) in a TH2I with a set of bins
    for each channel.

    Parameters
    ----------
    path: str
        The base path to the input ROOT file.
    run: int
        The run number of the input ROOT file.
    title: str
        The title to place on the plot.
    channel: int
        The channel number to retrieve and plot.
    scale: float
        The y-range of the plot. The range will be set to (-250, scale).
    """
    plt.style.use('../plot_style.mplstyle')
    
    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()
    peak_val = []
    integral = []
    fwhm = []
    data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
    for i in range(channel_start,channel_end):
        
        
        waveform, count = data[:-1, i].astype(float), data[-1,i]
        waveform -= np.median(waveform)
        waveform /= count

        pwaveform = waveform[:150]
        mpeak = np.argmin(waveform)
        mwaveform = -1 * waveform[mpeak-75:mpeak+75]
        peak = np.argmax(pwaveform)

        popt, pcov = curve_fit(bessel, np.arange(47),pwaveform[peak-7:peak+40],maxfev =50000)
        #print(pwaveform)
        print(popt)

        fit = bessel(np.arange(67),*popt)
        print("Peak value", fit[np.argmax(fit)])
        peak_val.append(fit[np.argmax(fit)])
        #options={'limit':100}

        result = integrate.quad(lambda x: bessel(x,*popt), -100, 2000,limit = 2000)
        print("Integral of fit from 0 to 2000 ", result) 
        integral.append(result[0])
        
        peaks, _ = find_peaks(fit)
        results_half = peak_widths(fit, peaks, rel_height=0.5)
        print("Peaks",peaks)
        print("Widths of peak in fit ",results_half[0])
        fwhm.append(results_half[0][0])
    
    
    #ax.scatter(range(channel_start,channel_end), integral,linewidth=5, label='Integral of fit -100 to 2000')
    #ax.scatter(range(channel_start,channel_end), peak_val,linewidth=5, label='Peaks of channels')
    ax.scatter(range(channel_start,channel_end), fwhm,linewidth=5, label="FWHM of channel's peak")
    #ax.plot(np.arange(67), pwaveform[peak-7:peak+60], linestyle='-', linewidth=2, label='Positive Lobe')
    #ax.plot(np.arange(150), mwaveform, linestyle='-', linewidth=2, label='Negative Lobe')
    ax.set_xlim(30, 70)
    #ax.set_ylim(-250, scale)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Width [Ticks]')
    ax.legend()
    figure.suptitle(title)
    
def fit_milind(path, run, title, channel_1=0, scale=2200,cross_talk = False):
    """
    Plots the average waveform for the specified channel using a ROOT
    file prepared by the TPCTestPulseArtDAQ module. This module aligns
    waveforms and stores them (as a sum) in a TH2I with a set of bins
    for each channel.

    Parameters
    ----------
    path: str
        The base path to the input ROOT file.
    run: int
        The run number of the input ROOT file.
    title: str
        The title to place on the plot.
    channel: int
        The channel number to retrieve and plot.
    scale: float
        The y-range of the plot. The range will be set to (-250, scale).
    """
    if cross_talk == True:
        channel_2 = int(input("Enter the other channel:"))
        plt.style.use('../plot_style.mplstyle')

        data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
        waveform_1, count_1 = data[:-1, channel_1].astype(float), data[-1,channel_1]
        waveform_1 -= np.median(waveform_1)
        waveform_1 /= count_1
        waveform_2, count_2 = data[:-1, channel_2].astype(float), data[-1,channel_2]
        waveform_2 -= np.median(waveform_2)
        waveform_2 /= count_2
        waveform = np.subtract(waveform_1,waveform_2)
    else:
        data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
        waveform, count = data[:-1, channel_1].astype(float), data[-1,channel_1]
        waveform -= np.median(waveform)
        waveform /= count
        
    plt.style.use('../plot_style.mplstyle')
    
    figure = plt.figure(figsize=(8,6))
    ax = figure.add_subplot()
    
    pwaveform = waveform[:150]
    mpeak = np.argmin(waveform)
    mwaveform = -1 * waveform[mpeak-75:mpeak+75]
    peak = np.argmax(pwaveform)
    print(peak)
    
    popt, pcov = curve_fit(milind, np.arange(0,46,1),pwaveform[peak-6:peak+40],bounds=(0., [.7, 1.5, np.inf,np.inf]),maxfev =50000)
    #print(pwaveform)
    print(popt,pcov)
    perr = np.sqrt(np.diag(pcov))
    print(perr)
    
    fit = milind(np.arange(0,47,.1),*popt)
    #print(fit)
    options={'limit':100}
    result = integrate.quad(lambda x: milind(x,*popt), 0, 2000,limit = 2000)
    peaks, _ = find_peaks(fit)
    results_half = peak_widths(fit, peaks, rel_height=0.5)
    diff = np.subtract(pwaveform[peak-6:peak+40], milind(np.arange(0,46,1),*popt))
    #print(diff)
    ratio = np.divide(diff**2, milind(np.arange(46),*popt))
    #print(ratio)
    #print(np.sum(ratio))
    chi_square = np.sum(np.divide((np.subtract(pwaveform[peak-6:peak+40], milind(np.arange(46),*popt))**2),milind(np.arange(46),*popt)))
    
    #data_result = integrate.quad(lambda x: pwaveform, 0, 100,limit = 2000)
    data_peaks, _ = find_peaks(pwaveform)
    data_results_half = peak_widths(pwaveform, data_peaks, rel_height=0.5)
    print("Fit statistics")
    print("Chi^2: ", chi_square)
    print("Fit Peaks",peaks/10)
    print("Fit Widths of peak in fit ",results_half[0]/10)
    print("Integral of fit from -100 to 2000 ", result) 
    print("Peak value", fit[np.argmax(fit)])
    
    print("Data statistics")
    print("Data Peaks",data_peaks)
    print("Data Widths of peak ",results_half[0])
    #print("Integral of fit from 0 to 100 ", data_result) 
    print("Data peak value", pwaveform[np.argmax(pwaveform)])
    
    ax.plot(np.arange(-10,56,.1), milind(np.arange(-10,56,.1),*popt), linestyle = 'dotted', linewidth=5, label="Milind's general fit")
    ax.plot(np.arange(56), pwaveform[peak-6:peak+50], linestyle='-', linewidth=2, label='Positive Lobe')
    #ax.plot(np.arange(150), mwaveform, linestyle='-', linewidth=2, label='Negative Lobe')
    #ax.set_xlim(15, 80)
    ax.set_xlim(-10, 60)
    ax.set_ylim(-250, scale)
    #ax.set_ylim(-250, 200)
    #ax.set_ylim(-50, 50)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    ax.legend()
    figure.suptitle(title)
def plot_diff_waveform_artdaq(path, run, title, channel_1=0, channel_2 = 32, scale=2200):
    """
    Plots the average waveform for the specified channel using a ROOT
    file prepared by the TPCTestPulseArtDAQ module. This module aligns
    waveforms and stores them (as a sum) in a TH2I with a set of bins
    for each channel.

    Parameters
    ----------
    path: str
        The base path to the input ROOT file.
    run: int
        The run number of the input ROOT file.
    title: str
        The title to place on the plot.
    channel: int
        The channel number to retrieve and plot.
    scale: float
        The y-range of the plot. The range will be set to (-scale, scale).
    """
    plt.style.use('../plot_style.mplstyle')
    
    figure = plt.figure(figsize=(16,6))
    ax = figure.add_subplot()
    
    data = uproot.open(f'{path}waveforms_run{run}.root')['tpctestpulseartdaq/output'].values()
    waveform_1, count_1 = data[:-1, channel_1].astype(float), data[-1,channel_1]
    waveform_1 -= np.median(waveform_1)
    waveform_1 /= count_1
    waveform_2, count_2 = data[:-1, channel_2].astype(float), data[-1,channel_2]
    waveform_2 -= np.median(waveform_2)
    waveform_2 /= count_2
    waveform = np.subtract(waveform_1,waveform_2)
    ax.plot(np.arange(len(waveform)), waveform, linestyle='-', linewidth=2,label = 'Difference')
    ax.plot(np.arange(len(waveform_1)), waveform_1, linestyle='-', linewidth=2, label = f'Channel {channel_1}')
    ax.plot(np.arange(len(waveform_2)), waveform_2, linestyle='-', linewidth=2,label = f'Channel {channel_2}')
    ax.set_xlim(0, len(waveform))
    ax.set_ylim(-scale, scale)
    ax.set_xlabel('Time [ticks]')
    ax.set_ylabel('Waveform Height [ADC]')
    ax.legend()
    figure.suptitle(title)