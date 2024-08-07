# %% ('cell' in vscode - interactive window)
import aotools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedLocator
from collections import OrderedDict
import tifffile

from skimage.restoration import unwrap_phase
from zern_utils import within_CTF
from pathlib import Path

plt.set_cmap("cividis")
plt.rcParams.update({'font.size': 18})


def Noll2OSA_OSA2Noll(order_num0):
    """
    transform zernike amplitudes, from Noll's index -> to OSA/ANSI
    [Na Ji's paper](https://doi.org/10.1038/ncomms8276) follows OSA/ANSI index while AOTools Use Noll's index.  
    See [wikipedia-zernike polynomials](https://en.wikipedia.org/wiki/Zernike_polynomials) for explanations of these indices.
    """

    max_n, _ = aotools.functions.zernike.zernIndex(order_num0)
    order_num = (max_n+1)*(max_n+2)//2 # consider all zernike polynomials of which n=n_max.

    noll_nms=OrderedDict()
    noll2osa=OrderedDict()
    for j in range(1, order_num+1):
        noll_nms[j]=aotools.functions.zernike.zernIndex(j)
        n, m = noll_nms[j]
        noll2osa[j]=(n*(n+2)+m)//2

    osa2noll=OrderedDict(zip(noll2osa.values(), noll2osa.keys()))
    osa2noll = dict(sorted(osa2noll.items()))
    return noll2osa, osa2noll

def zern_bases_OSAidx(order_num0, pupil_size):
    """
    return zern_bases, re-ordered to OSA index
    """
    _, osa2noll = Noll2OSA_OSA2Noll(order_num0)
    zern_bases = aotools.zernikeArray(len(osa2noll), pupil_size)
    zern_bases_OSAindex = np.zeros_like(zern_bases)
    for i in range(order_num0):
        zern_bases_OSAindex[i]=zern_bases[osa2noll[i]-1]
    return zern_bases_OSAindex

def plot_zern_amp(zern_amp, num_modes, label=None, ylim=None, savefig=None):
    plt.close()
    fig, ax = plt.subplots(figsize=(12,6))
    zern_amp= np.array(zern_amp, dtype=float)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(FixedLocator(list(range(round(zern_amp.min()-1), round(zern_amp.max()+1)))))
    ax.yaxis.set_minor_locator(FixedLocator(list(np.arange(round(zern_amp.min()-1), round(zern_amp.max()+1), 0.2))))
    #ax.yaxis.set_major_locator()
    plt.xticks(list(range(0, num_modes+1, 5)))
    plt.xlim([-0.5, float(num_modes)+0.5])
    ax.grid(axis='x', color='0.85', ls="--")
    ax.grid(axis='y', color='0.90', ls="--", which="both")
    if ylim!=None:
        plt.ylim(ylim)
    elif np.max(np.abs(zern_amp))<1.0:
        plt.ylim([-1.0, 1.0])
    if np.shape(zern_amp)==(2, num_modes):
        ax.plot(list(range(num_modes)), zern_amp[0], 'bs', label = label[0]) 
        ax.plot(list(range(num_modes)), zern_amp[1], 'r.', label = label[1]) 
        ax.legend()
    else:
        ax.bar(list(range(num_modes)), zern_amp, label = label) # preview zern_amps
        ax.legend()
    
    if savefig:
        fig.savefig(savefig)
    else:
        plt.show()
    plt.close(fig)

def decompose2zern_amp(img, num_order):
    num_order = num_order
    image_unwrapped_wrap_around = unwrap_phase(img, wrap_around=(True, False))
    zern_bases = zern_bases_OSAidx(num_order, len(img))
    area = np.pi*len(img)**2/4.0
    return [np.sum(zern_bases[i]*image_unwrapped_wrap_around)/area for i in range(num_order)]

def cancel_defocus(phasemap, wrapped = True, num_order = 60, zern_idx2compensate = (0, 4, 12, 24)):
    """
    cancel 0, 4, 12, 24 modes of zernike bases
    Should pass CTF_size=1.0 phasemaps. (full circle on the square sized image)
    """
    is_in_circle = within_CTF(len(phasemap), 1.0).numpy()
    phase_size = len(phasemap)
    area = np.pi*phase_size**2/4.0
    
    mask = np.logical_not(is_in_circle)
    zern_bases = zern_bases_OSAidx(num_order, phase_size)

    image_unwrapped = unwrap_phase(phasemap, wrap_around=(True, False))

    decompose_zern_amp = [np.sum(zern_bases[i]*image_unwrapped)/area for i in range(num_order)]

    cancel_amp = [decompose_zern_amp[i] if (i in zern_idx2compensate) else 0 for i in range(num_order)]
    cancel_phasemap = np.sum([cancel_amp[i] * zern_bases[i] for i in  range(num_order)], axis=0)

    return np.ma.getdata(image_unwrapped-cancel_phasemap, subok=False)*is_in_circle

def convert_tif2canceled_defocus(from_dir, to_dir):
    """
    read wrapped phasemap, convert and save
    """
    for from_file in Path(from_dir).iterdir():
        mod = tifffile.imread(from_file)
        mod_no_defocus = cancel_defocus(mod)
        to_file = Path(to_dir).joinpath(f"{from_file.stem}.tif")
        tifffile.imwrite(to_file, mod_no_defocus)

def phasemap_from_zern_amp(zern_amp, phasemap_size, return_wrapped=False):
    num_order = len(zern_amp)
    zern_bases = zern_bases_OSAidx(num_order, phasemap_size)
    phasemap = np.sum([zern_amp[i] * zern_bases[i] for i in range(num_order)], axis=0)
    return phasemap, np.angle(np.exp(1j*phasemap))

def phasemap_vs_phasemap(phasemap1=None, subtitle1='1', phasemap2=None,  subtitle2='2', savefig=None):
    fig, (ax1, ax2) =plt.subplots(1, 2, figsize=(19, 9))
    im = ax1.imshow(phasemap1)
    fig.colorbar(im, ax=ax1)
    ax2.axis('off')
    ax1.set_title(subtitle1)
    im = ax2.imshow(phasemap2)
    fig.colorbar(im, ax=ax2)
    ax2.axis('off')
    ax2.set_title(subtitle2)
    if savefig:
        fig.savefig(savefig)
    plt.close(fig)
        

def compare_final_abrr_vs_input_abrr(pth_final_abrr, fixed_abrr_amp, post_output_dir, 
fixed_aberration_type=None):
    if fixed_aberration_type==None:
        assert type(fixed_abrr_amp) is str
        fixed_aberration_type = fixed_abrr_amp
    if fixed_abrr_amp == "low":
        fixed_abrr_amp = [0,0,0,0.200000000000000,0,-0.200000000000000,0.500000000000000,0.500000000000000,-0.300000000000000,0.300000000000000,0.100000000000000,0.500000000000000,0.100000000000000,0.200000000000000,0.300000000000000]
    elif fixed_abrr_amp == "high":
        fixed_abrr_amp = [0,0,0,0.700000000000000,0,-1.40000000000000,-1.10000000000000,1.10000000000000,-0.200000000000000,-0.300000000000000,1.40000000000000,0.800000000000000,1.50000000000000,-0.800000000000000,-1.30000000000000]
    elif fixed_abrr_amp == "naji":
        fixed_abrr_amp = [0,0,0,0.100000000000000,0,-0.200000000000000,0.200000000000000,-0.400000000000000,1.20000000000000,-0.800000000000000,0.250000000000000,-0.100000000000000,-0.600000000000000,0.100000000000000,0.700000000000000,0,0.100000000000000,-0.100000000000000,-0.600000000000000,-0.600000000000000,-0.400000000000000,0.400000000000000,0.100000000000000,0,0,0,0.100000000000000,0.800000000000000,-0.200000000000000,0,-0.100000000000000,-0.100000000000000,-0.200000000000000,0.100000000000000,-0.100000000000000,0.150000000000000,0.200000000000000,0.100000000000000,-0.100000000000000,0.100000000000000,-0.100000000000000,0,0.150000000000000,0.150000000000000,-0.100000000000000,0,-0.100000000000000,0,0,0.100000000000000,0.100000000000000,0.100000000000000,0,0.100000000000000,0.300000000000000,0,0,0,0,0]
    elif type(fixed_abrr_amp) is not list:
        raise NotImplementedError
    final_abrr = tifffile.imread(pth_final_abrr)
    assert final_abrr.shape[0]==final_abrr.shape[1]
    fixed_abrr_map = phasemap_from_zern_amp(fixed_abrr_amp, final_abrr.shape[0])
    phasemap_vs_phasemap(np.angle(np.exp(1j*cancel_defocus(final_abrr))), "final aberration(defocus canceled)", np.angle(np.exp(1j*fixed_abrr_map[0])), f"{fixed_aberration_type} aberration", savefig = Path(post_output_dir).joinpath(f"final_abrr(defocus_canceled)-{fixed_aberration_type}_fixed_abrr.png"))
    phasemap_vs_phasemap(np.angle(np.exp(1j*final_abrr)), "final aberration", np.angle(np.exp(1j*fixed_abrr_map[0])), f"{fixed_aberration_type} aberration", savefig = Path(post_output_dir).joinpath(f"final_abrr-{fixed_aberration_type}_fixed_abrr.png"))
    phasemap_vs_phasemap(np.angle(np.exp(1j*cancel_defocus(final_abrr))), "final aberration(defocus canceled)", np.angle(np.exp(1j*cancel_defocus(fixed_abrr_map[0]))), f"{fixed_aberration_type} aberration(defocus canceled)", savefig = Path(post_output_dir).joinpath(f"final_abrr(defocus_canceled)-{fixed_aberration_type}_fixed_abrr(defocus_canceled).png"))
    plot_zern_amp((decompose2zern_amp(cancel_defocus(final_abrr), 60), decompose2zern_amp(cancel_defocus(fixed_abrr_map[0]), 60)), 60, label=("final abrr(defocus canceled)", "input abrr"), savefig = Path(post_output_dir).joinpath(f"zern_amp_comp_final_abrr(defocus_canceled)-{fixed_aberration_type}_fixed_abrr.png"))
    plot_zern_amp((decompose2zern_amp(final_abrr, 60), decompose2zern_amp(fixed_abrr_map[0], 60)), 60, label=("final abrr", "input abrr"), savefig = Path(post_output_dir).joinpath(f"zern_amp_comp_final_abrr-{fixed_aberration_type}_fixed_abrr.png"))
    
# %% Validation / instruction
if __name__ == "__main__":
    # plot zern_amplitude by laji group's paper 
    # [Wang, Kai, Wenzhi Sun, Christopher T. Richie, Brandon K. Harvey, Eric Betzig, and Na Ji. “Direct Wavefront Sensing for High-Resolution in Vivo Imaging in Scattering Tissue.” Nature Communications 6, no. 1 (June 15, 2015): 7276.](https://doi.org/10.1038/ncomms8276.)
    num_order = 60
    noll2osa, osa2noll = Noll2OSA_OSA2Noll(num_order)

    zern_amp=[0,0,0,0.100000000000000,0,-0.200000000000000,0.200000000000000,-0.400000000000000,1.20000000000000,-0.800000000000000,0.250000000000000,-0.100000000000000,-0.600000000000000,0.100000000000000,0.700000000000000,0,0.100000000000000,-0.100000000000000,-0.600000000000000,-0.600000000000000,-0.400000000000000,0.400000000000000,0.100000000000000,0,0,0,0.100000000000000,0.800000000000000,-0.200000000000000,0,-0.100000000000000,-0.100000000000000,-0.200000000000000,0.100000000000000,-0.100000000000000,0.150000000000000,0.200000000000000,0.100000000000000,-0.100000000000000,0.100000000000000,-0.100000000000000,0,0.150000000000000,0.150000000000000,-0.100000000000000,0,-0.100000000000000,0,0,0.100000000000000,0.100000000000000,0.100000000000000,0,0.100000000000000,0.300000000000000,0,0,0,0,0] # naji group, bio-sample aberration
    plot_zern_amp(zern_amp, num_order)
    # %% validation of zern_bases in osaidx
    pupil_size = round(256*0.45)
    zern_bases = zern_bases_OSAidx(num_order, pupil_size)
    fig, ax = plt.subplots(5, 6)
    for i in range(5):
        for j in range(6):
            ax[i, j].imshow(zern_bases[j+6*i])
            ax[i, j].axis('off')
            ax[i,j].set_title(str(j+6*i))

    # %% plot phasemap from laji group's zern_amp
    naji_phasemaps = [zern_amp[i] * zern_bases[i] for i in range(num_order)]
    naji_phasemap = np.sum(naji_phasemaps, axis=0)
    fig, (ax1, ax2) =plt.subplots(1, 2, figsize=(7, 3))
    im = ax1.imshow(naji_phasemap)
    fig.colorbar(im, ax=ax1)
    wrap_naji_phasemap = np.angle(np.exp(1j*naji_phasemap))
    im = ax2.imshow(wrap_naji_phasemap)
    fig.colorbar(im, ax=ax2)
    ax2.axis('off')

    # %% validate zern_bases normalization:
    area = np.pi*pupil_size*pupil_size/4.0 # area of circle
    delta = [np.sum(zern_bases[i]*zern_bases[i])/area for i in range(num_order)] #Kronecker Delta...
    plot_zern_amp(delta, num_order)

    # %% validate zern_bases orthogonality:
    assert num_order>=60
    delta = np.zeros((60, 60))
    for j in range(60):
        delta[j] = [np.abs(np.sum(zern_bases[i]*zern_bases[j]))/area for i in range(60)]
        #plt.bar(list(range(60-j)), unity) # preview zern_amps
        #print(f"'integration of (zernike_basis[i], zernike_basis[i+k])/area' have very samll values... {np.max(np.abs(unity)):.014f} where j: {j}")
    plt.imshow(delta)
    plt.colorbar()
    # %% decompose zern_amp : pupil size 256*0.45 seems big enough...
    decompose_zern_amp = [np.sum(zern_bases[i]*naji_phasemap)/area for i in range(60)]
    plot_zern_amp(decompose_zern_amp, num_order)
    # is expected to be same with 'zern_amp'
    
    # %% pupil size = 256*0.4 seems big enough.
    plot_zern_amp((decompose_zern_amp, zern_amp), num_order) 
    # %% it is expected to be same with input zernike amplitude
    naji_phasemap_from_decompose = np.sum([decompose_zern_amp[i] * zern_bases[i] for i in range(num_order)], axis=0)
    plt.imshow(naji_phasemap_from_decompose); plt.colorbar()

    # %% From now on, Let's compensate specific zernike modes from phasemap.

    #preview defocuses...
    zern_idx2compensate = (4, 12, 24)
    zern_idx2compensate_amp = [1.0 if (i in zern_idx2compensate) else 0.0 for i in range(num_order)]
    defocuses_val_maps = [zern_idx2compensate_amp[i] * zern_bases[i] for i in range(num_order)]
    defocuses_val_map = np.sum(defocuses_val_maps, axis=0)
    plt.imshow(defocuses_val_map);plt.colorbar()

    # %%
    cancel_amp = [decompose_zern_amp[i] if (i in zern_idx2compensate) else 0 for i in range(num_order)]
    cancel_phasemap = np.sum([cancel_amp[i] * zern_bases[i] for i in  range(num_order)], axis=0)
    plot_zern_amp((cancel_amp, zern_amp), num_order) 
    # %%
    canceled_naji_phasemap = naji_phasemap-cancel_phasemap
    fig, (ax1, ax2) =plt.subplots(1, 2, figsize=(7, 3))
    im = ax1.imshow(canceled_naji_phasemap)
    fig.colorbar(im, ax=ax1)
    wrap_canceled_naji_phasemap = np.angle(np.exp(1j*canceled_naji_phasemap))
    im = ax2.imshow(wrap_canceled_naji_phasemap)
    fig.colorbar(im, ax=ax2)
    ax2.axis('off')
    # %% decomposition: spherical aberration effectively canceled.
    decompose_zern_amp = [np.sum(zern_bases[i]*canceled_naji_phasemap)/area for i in range(60)]
    plot_zern_amp((decompose_zern_amp,zern_amp), num_order)

    #%%
    fig, (ax1, ax2) =plt.subplots(1, 2, figsize=(9, 4))
    im = ax1.imshow(wrap_naji_phasemap)
    fig.colorbar(im, ax=ax1)
    ax2.axis('off')
    ax1.set_title("Original Na Ji - phasemap")
    im = ax2.imshow(wrap_canceled_naji_phasemap)
    fig.colorbar(im, ax=ax2)
    ax2.axis('off')
    ax2.set_title("defocus compensated")

    # %% Let's do this on our phasemaps. this one looks like have large defocus...
    # datasets/modulation_swcho_0725/2/2_78.tif
    phasemap = tifffile.imread(r"datasets/modulation_swcho_0725/2/2_78.tif")
    plt.imshow(phasemap);plt.colorbar()

    # %% first, it should be unwrapped. Let's run a example code of scikit-image, [Phase Unwrapping](https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html)

    is_in_circle = within_CTF(len(phasemap), 1.0)
    mask = np.logical_not(is_in_circle)

    from skimage.restoration import unwrap_phase
    from zern_utils import within_CTF
    is_in_circle = within_CTF(len(phasemap), 1.0)
    mask = np.logical_not(is_in_circle)

    image_wrapped = np.ma.array(np.angle(np.exp(1j * phasemap)), mask=mask)
    # Unwrap image without wrap around
    image_unwrapped_no_wrap_around = unwrap_phase(image_wrapped, wrap_around=(False, False))
    # Unwrap with wrap around enabled for the 0th dimension
    image_unwrapped_wrap_around = unwrap_phase(image_wrapped, wrap_around=(True, False))

    fig, ax = plt.subplots(2, 2)
    ax1, ax2, ax3, ax4 = ax.ravel()

    fig.colorbar(ax1.imshow(np.ma.array(phasemap, mask=mask), cmap='rainbow'), ax=ax1)
    ax1.set_title('Original')

    fig.colorbar(ax2.imshow(image_wrapped, cmap='rainbow', vmin=-np.pi, vmax=np.pi), ax=ax2)
    ax2.set_title('Wrapped phase')

    fig.colorbar(ax3.imshow(image_unwrapped_no_wrap_around, cmap='rainbow'), ax=ax3)
    ax3.set_title('Unwrapped without wrap_around')

    fig.colorbar(ax4.imshow(image_unwrapped_wrap_around, cmap='rainbow'), ax=ax4)
    ax4.set_title('Unwrapped with wrap_around')

    plt.tight_layout()
    plt.show()
    # %%

    decompose_zern_amp = [np.sum(zern_bases[i]*image_unwrapped_wrap_around)/area for i in range(60)]

    zern_idx2compensate = (0, 4, 12, 24)
    cancel_amp = [decompose_zern_amp[i] if (i in zern_idx2compensate) else 0 for i in range(num_order)]
    cancel_phasemap = np.sum([cancel_amp[i] * zern_bases[i] for i in  range(num_order)], axis=0)
    plot_zern_amp((cancel_amp, decompose_zern_amp), num_order) 
    # %%
    canceled_phasemap = image_unwrapped_wrap_around-cancel_phasemap
    fig, (ax1, ax2) =plt.subplots(1, 2, figsize=(7, 3))
    im = ax1.imshow(canceled_phasemap)
    fig.colorbar(im, ax=ax1)
    wrap_canceled_phasemap = np.angle(np.exp(1j*canceled_phasemap))
    im = ax2.imshow(wrap_canceled_phasemap)
    fig.colorbar(im, ax=ax2)
    ax2.axis('off')
    # %% decomposition: spherical aberration effectively canceled.
    canceled_decompose_zern_amp = [np.sum(zern_bases[i]*canceled_phasemap)/area for i in range(60)]
    plot_zern_amp((canceled_decompose_zern_amp,decompose_zern_amp), num_order)

    # %%
