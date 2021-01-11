import numpy as np
from builtins import sum as bsum
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cycler, cm
from matplotlib import colors as mcolors

cols={0.01:mcolors.to_rgba("mediumseagreen"), 0.025:cm.gray(0.25)}

def setup_axes(ax, title_size = 12, xlabel_size=10, ylabel_size=10, xticklabel_size=8, yticklabel_size=8):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    items = [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    sizes = [title_size, xlabel_size, ylabel_size] + [xticklabel_size]*len(ax.get_xticklabels()) + [yticklabel_size]*len(ax.get_yticklabels())
             
    for (item, sz) in zip(items, sizes):
        item.set_fontsize(sz)

def set_alpha(c, a):
    return list(c[:3]) + [a]

def plot_single_osn_response(ax, V, stim, n_trials, t_pre, dt, max_trial_show = 20, trace_col = "lightgray", stim_offset = None, stim_scale = None):
    Vr = np.reshape(V[:,0],(n_trials,-1)).T
    tt    = np.arange(Vr.shape[0])*dt
    Vrm   = np.mean(Vr,axis=1)
    scale = np.max(Vrm)
    stim_offset = stim_offset if stim_offset else np.max(Vr) + 0.1*scale
    stim_scale  = stim_scale  if stim_scale  else scale
    ax.plot(tt - t_pre, Vr[:, :max_trial_show], color=trace_col)
    ax.plot(tt - t_pre, Vrm,"r",linewidth=1)
    ax.plot(tt - t_pre, stim[:len(Vr)]*stim_scale+stim_offset,"k",linewidth=0.5)
    return ax
    #ax.set_ylim(0, np.max(Vr))


def setup_axes_for_poster(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.xaxis.set_tick_params(length=2, which="major")
    ax.yaxis.set_tick_params(length=2, which="major")
    ax.xaxis.set_tick_params(length=1, which="minor")
    ax.yaxis.set_tick_params(length=1, which="minor")
    ax.tick_params(labelsize=10)

def str2stim(s, widths, t_trial = 1, n_trials = 1, dt=1):
    stim = []
    n_trial = int(round(t_trial/dt))
    for item in s:
        if item != "!":
            stim.append([int(item.isupper())]*int(round(widths[item]/dt)))

    stim = bsum(stim, [])
    if s[-1] == "!":
        stim += [0]*(n_trial - len(stim))
    stim = stim*n_trials
    t = np.arange(len(stim))*dt
    return np.array(stim), t
    

def run(stim, amp, n_osn=100, tau_chem = 0.4, tau_osn = 0.1, sd = 0.5, th = 1, v_ref = -1, dt = 1e-4, t_ref = 1, seed = 0, **kwargs):
    np.random.seed(seed)
    
    t    = np.arange(len(stim))*dt
    nt   = len(t)
    stim = np.array(stim)*amp
    c    = np.zeros(nt, ) # The olfactory current

    V    = np.random.randn(nt, n_osn)*sd # The membrane voltages
    V[0] = 0 # Initialize to zero

    S    = 0*V # Spike indicator
    
    ind    = np.arange(n_osn)
    spikes = []
    last_spike = -np.inf + np.zeros(n_osn,) # Last spike times at t = -inf
    for i in range(1, nt):
        # tau_chem dc/dt = -c + stim
        dc = (-c[i-1] + stim[i-1])*dt/tau_chem
        c[i] = c[i-1] + dc

        # tau dv = (-v + c) dt + dW(t)        
        dv = ((-V[i-1] + c[i-1])*dt + np.sqrt(dt)*V[i])/tau_osn # V[i] is noise
        dv *= (t[i] - last_spike)>=t_ref # Only update voltage if you've been in refractory long enough
        V[i] = V[i-1] + dv
        V[i][spikes] = v_ref # Set the voltages of those who spiked last time step to ref
        spikes       = ind[V[i]>=th] # Index of those whose spiked this time step
        S[i, spikes]  = 1
        last_spike[spikes] = t[i]
    return V, S, c

