"""!
@file
File containing functions for generating plots.
"""

import numpy as np
import matplotlib.pyplot as pt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import warnings
from traceback import print_tb
warnings.filterwarnings("ignore")

import PyPO.PlotConfig
import PyPO.Colormaps as cmaps
import PyPO.BindRefl as BRefl
from PyPO.Enums import Projections, FieldComponents, CurrentComponents, Units, Scales

def plotBeam2D(plotObject, field, contour,
                vmin, vmax, levels, amp_only,
                norm, aperDict, scale, project,
                units, title, titleA, titleP, unwrap_phase, correct_phase=False, k=None):
    """!
    Generate a 2D plot of a field or current.

    @param plotObject A reflDict containing surface on which to plot beam. 
    @param field PyPO field or current component to plot.
    @param contour A PyPO field or current component to plot as contour.
    @param vmin Minimum amplitude value to display. Default is -30.
    @param vmax Maximum amplitude value to display. Default is 0.
    @param levels Levels for contourplot.
    @param amp_only Only plot amplitude pattern. Default is False.
    @param norm Normalise field (only relevant when plotting linear scale).
    @param aperDict Plot an aperture defined in an aperDict object along with the field or current patterns. Default is None.
    @param scale Plot amplitude in decibels, logarithmic or linear scale. Instance of Scales enum object.
    @param project Set abscissa and ordinate of plot. Should be given as an instance of the Projection enum.
    @param units The units of the axes. Instance of Units enum object.
    @param title An overall title for the plot. Defaults to the field name and component.
    @param titleA Title of the amplitude plot. Default is "Amp".
    @param titleP Title of the phase plot. Default is "Phase".
    @param unwrap_phase Unwrap the phase pattern. Prevents annular structure in phase pattern. Default is False.
    @param correct_phase Boolean or 3 element numpy array. Applies a phase factor to the field equal to
            k*displacement of the grid along the Z-axis (True) or direction of the 3-vector.
    @param k Wavenumber to use for phase correction. Only used if correct_phase is not False.

    @returns fig Figure object containing plot.
    @returns ax Axes containing the axes of the plot.

    @see aperDict
    """

    # With far-field, generate grid without converting to spherical
    max_field = np.max(np.absolute(field))
    grids = BRefl.generateGrid(plotObject, transform=True, spheric=False)
    if not plotObject["gmode"] == 2:
        if units.dimension != 'spatial':
            units = Units.MM
        if project == Projections.xy:
            grid_x1 = grids.x
            grid_x2 = grids.y
            ff_flag = False
            comps = ["x", "y"]

        elif project == Projections.yz:
            grid_x1 = grids.y
            grid_x2 = grids.z
            ff_flag = False
            comps = ["y", "z"]

        elif project == Projections.zx:
            grid_x1 = grids.z
            grid_x2 = grids.x
            ff_flag = False
            comps = ["z", "x"]

        elif project == Projections.yx:
            grid_x1 = grids.y
            grid_x2 = grids.x
            ff_flag = False
            comps = ["y", "x"]

        elif project == Projections.zy:
            grid_x1 = grids.z
            grid_x2 = grids.y
            ff_flag = False
            comps = ["z", "y"]

        elif project == Projections.xz:
            grid_x1 = grids.x
            grid_x2 = grids.z
            ff_flag = False
            comps = ["x", "z"]

    else: # probably a farfield grid
        if units.dimension != 'angular':
            units = Units.DEG
            
        if project == Projections.xy:
            grid_x1 = grids.x
            grid_x2 = grids.y
            ff_flag = True
            comps = ["\mathrm{Az}", "\mathrm{El}"]

        elif project == Projections.yx:
            grid_x1 = grids.y
            grid_x2 = grids.x
            ff_flag = True
            comps = ["\mathrm{El}", "\mathrm{Az}"]

        if not ((units == Units.DEG) or (units == Units.AM) or (units == Units.AS)):
            units = Units.DEG
    if not amp_only:
        fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.5})

        if correct_phase is not False:
            if type(correct_phase) is bool:
                correct_phase = 1
                
            if type(correct_phase) is int or type(correct_phase) is float:
                correct_phase = int(np.sign(correct_phase))
                # Correct phase for z-axis displacement
                if plotObject['gmode'] == 1:
                    correct_phase = correct_phase*np.array((np.mean(grids.nx[0,:]), np.mean(grids.ny[0,:]), np.mean(grids.nz[0,:])))
                    vnorm = correct_phase / np.linalg.vector_norm(correct_phase)
                elif plotObject['gmode'] == 0:
                    shape = grids.z.shape
                    n,m = int(shape/2)
                    vnorm = correct_phase*np.array((grids.nx[n,m], grids.ny[n,m], grids.nz[n, m]))
            else: # Correct_phase is a vector
                try:
                    if len(correct_phase) != 3:
                        raise ValueError
                except (ValueError, KeyError, TypeError):
                    raise ValueError("correct_phase must be either boolean, number or np.ndarray((nx, ny, nz))")
                vnorm = correct_phase / np.linalg.vector_norm(correct_phase)
            
            offset = np.linalg.vecdot(vnorm, np.stack((grids.x, grids.y, grids.z), axis=-1))

            if plotObject['gmode'] == 1:
                r0 = np.mean(offset[0,:])
                phase_factor = np.exp(-1j*k*(offset-r0))
            elif plotObject['gmode'] == 0:
                r0 = offset[n,m]
                phase_factor = np.exp(-1j*k*(offset-r0))
            else: # Don't know what to do for farfields
                phase_factor = 1.0
        else:
            phase_factor = 1.0


        if scale == Scales.LIN:
            if norm:
                field_pl = np.absolute(field) / max_field
                if contour is not None:
                    contour = np.absolute(contour) / np.max(np.absolute(contour))
            
            else:
                field_pl = np.absolute(field)
                if contour is not None:
                    contour = np.absolute(contour)

            vmax = np.nanmax(field_pl) if vmax is None else vmax
            if vmin is None:
                vmin = np.nanmin(field_pl)
                    
            if unwrap_phase:
                phase = np.unwrap(np.unwrap(np.angle(field*phase_factor), axis=0), axis=1)

            else:
                phase = np.angle(field*phase_factor)
            
            ampfig = ax[0].pcolormesh(grid_x1/units, grid_x2/units, field_pl**2,
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')
            phasefig = ax[1].pcolormesh(grid_x1/units, grid_x2/units, phase, cmap=cmaps.parula, shading='auto')

            if contour is not None:
                cont0 = ax[0].contour(grid_x1/units, grid_x2/units, contour**2, levels, cmap=cm.binary, linewidths=0.5)
                cont1 = ax[1].contour(grid_x1/units, grid_x2/units, np.angle(contour), levels, cmap=cm.binary, linewidths=0.5)

                ax[0].clabel(cont0)
                ax[1].clabel(cont1)

        if scale == Scales.AMP:
            if norm:
                field_pl = np.absolute(field) / max_field
                if contour is not None:
                    contour = np.absolute(contour) / np.max(np.absolute(contour))
            
            else:
                field_pl = np.absolute(field)
                if contour is not None:
                    contour = np.absolute(contour)

            vmax = np.nanmax(field_pl) if vmax is None else vmax
            if vmin is None:
                vmin = np.nanmin(field_pl)
            
            if unwrap_phase:
                phase = np.unwrap(np.unwrap(np.angle(field*phase_factor), axis=0), axis=1)

            else:
                phase = np.angle(field*phase_factor)
            
            ampfig = ax[0].pcolormesh(grid_x1 / units.value, grid_x2 / units.value, field_pl,
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')
            phasefig = ax[1].pcolormesh(grid_x1 / units.value, grid_x2 / units.value, phase, cmap=cmaps.parula, shading='auto')

            if contour is not None:
                cont0 = ax[0].contour(grid_x1 / units.value, grid_x2 / units.value, contour, levels, cmap=cm.binary, linewidths=0.5)
                cont1 = ax[1].contour(grid_x1 / units.value, grid_x2 / units.value, np.angle(contour), levels, cmap=cm.binary, linewidths=0.5)

                ax[0].clabel(cont0)
                ax[1].clabel(cont1)

        else: #  scale == Scales.dB
            if titleA == "Power":
                titleA += " (dB)"
            if titleP == "Phase":
                titleP += " (rad)"
                
            if norm:
                field_dB = 20 * np.log10(np.absolute(field) / max_field)
            else:
                field_dB = 20 * np.log10(np.absolute(field))
            
            if contour is not None:
                if norm:
                    contour_dB = 20 * np.log10(np.absolute(contour) / np.max(np.absolute(contour)))
                else:
                    contour_dB = 20 * np.log10(np.absolute(contour))
            
            vmax = np.nanmax(field_dB) if vmax is None else vmax
            if norm:
                vmin = np.nanmin(field_dB) if vmin is None else vmin
            else:
                vmin = np.nanmin(field_dB) if vmin is None else vmax - abs(vmin)
            
            if unwrap_phase:
                phase = np.unwrap(np.unwrap(np.angle(field*phase_factor), axis=0), axis=1)

            else:
                phase = np.angle(field*phase_factor)
            
            ampfig = ax[0].pcolormesh(grid_x1/units, grid_x2/units, field_dB,
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')
            phasefig = ax[1].pcolormesh(grid_x1/units, grid_x2/units, phase, cmap=cmaps.parula, shading='auto')
            

            if contour is not None:
                cont0 = ax[0].contour(grid_x1/units, grid_x2/units, contour_dB, levels, cmap=cm.binary, linewidths=0.5)
                cont1 = ax[1].contour(grid_x1/units, grid_x2/units, np.angle(contour), levels, cmap=cm.binary, linewidths=0.5)
                
                ax[0].clabel(cont0)
                ax[1].clabel(cont1)
        
        divider1 = make_axes_locatable(ax[0])
        divider2 = make_axes_locatable(ax[1])

        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)

        c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')
        c2 = fig.colorbar(phasefig, cax=cax2, orientation='vertical')

        ax[0].set_ylabel(r"${}$ ({})".format(comps[1], units.name))
        ax[0].set_xlabel(r"${}$ ({})".format(comps[0], units.name))
        ax[1].set_ylabel(r"${}$ ({})".format(comps[1], units.name))
        ax[1].set_xlabel(r"${}$ ({})".format(comps[0], units.name))

        ax[0].set_title(titleA, y=1.08)
        ax[0].set_aspect(1)
        ax[1].set_title(titleP, y=1.08)
        ax[1].set_aspect(1)


    else:
        fig, ax = pt.subplots(1,1, figsize=(5,5))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        if scale == Scales.LIN:
            if norm:
                field_pl = np.absolute(field) / max_field
                if contour is not None:
                    contour_pl = np.absolute(contour) / np.max(np.absolute(contour))
            
            else:
                field_pl = np.absolute(field)
                if contour is not None:
                    contour_pl = np.absolute(contour)

            vmin = np.min(field_pl) if vmin is None else vmin
            vmax = np.max(field_pl) if vmax is None else vmax
            
            ampfig = ax.pcolormesh(grid_x1/units, grid_x2/units, field_pl**2,
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')

            if contour is not None:
                cont = ax.contour(grid_x1/units, grid_x2/units, contour_pl**2, levels, cmap=cm.binary, linewidths=0.5)
                ax.clabel(cont)
        
        elif scale == Scales.dB:
            if titleA == "Power":
                titleA += " / dB"
            field_dB = 20 * np.log10(np.absolute(field) / max_field)
            
            if contour is not None:
                contour_dB = 20 * np.log10(np.absolute(contour) / np.max(np.absolute(contour)))
            
            vmin = np.min(field_dB) if vmin is None else vmin
            vmax = np.max(field_dB) if vmax is None else vmax
            
            ampfig = ax.pcolormesh(grid_x1/units, grid_x2/units, field_dB,
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')
            
            if contour is not None:
                cont = ax.contour(units.rdiv(grid_x1), units.rdiv(grid_x2), contour_dB, levels, cmap=cm.binary, linewidths=0.5)
                ax.clabel(cont)

        ax.set_ylabel(r"${}$ ({})".format(comps[1], units.name))
        ax.set_xlabel(r"${}$ ({})".format(comps[0], units.name))

        ax.set_title(titleA, y=1.08)
        ax.set_box_aspect(1)

        c = fig.colorbar(ampfig, cax=cax, orientation='vertical')
    
    if title is not None:
        fig.suptitle(title)
    
    if aperDict["plot"]:
        if aperDict["shape"] == "ellipse":
            xc = aperDict["center"][0]
            yc = aperDict["center"][1]
            Ro = 2*aperDict["outer"]
            Ri = 2*aperDict["inner"]


            if isinstance(ax, np.ndarray):
                for axx in ax:
                    circleo=mpl.patches.Ellipse((xc,yc),Ro[0], Ro[1], color='black', fill=False)
                    circlei=mpl.patches.Ellipse((xc,yc),Ri[0], Ri[1], color='black', fill=False)
                    
                    axx.add_patch(circleo)
                    axx.add_patch(circlei)
                    axx.scatter(xc, yc, color='black', marker='x')
            
            else:
                circleo=mpl.patches.Ellipse((xc,yc),Ro[0], Ro[1], color='black', fill=False)
                circlei=mpl.patches.Ellipse((xc,yc),Ri[0], Ri[1], color='black', fill=False)
                ax.add_patch(circleo)
                ax.add_patch(circlei)
                ax.scatter(xc, yc, color='black', marker='x')
        
        elif aperDict["shape"] == "rectangle":
            xco = aperDict["center"][0] + aperDict["outer_x"][0]
            yco = aperDict["center"][1] + aperDict["outer_y"][0]
            ho = aperDict["outer_y"][1] - aperDict["outer_y"][0]
            wo = aperDict["outer_x"][1] - aperDict["outer_x"][0]
            
            xci = aperDict["center"][0] + aperDict["inner_x"][0]
            yci = aperDict["center"][1] + aperDict["inner_y"][0]
            hi = aperDict["inner_y"][1] - aperDict["inner_y"][0]
            wi = aperDict["inner_x"][1] - aperDict["inner_x"][0]


            if isinstance(ax, np.ndarray):
                for axx in ax:
                    recto=mpl.patches.Rectangle((xco,yco),wo, ho, color='black', fill=False)
                    recti=mpl.patches.Rectangle((xci,yci),wi, hi, color='black', fill=False)
                    
                    axx.add_patch(recto)
                    axx.add_patch(recti)
                    axx.scatter(xco, yco, color='black', marker='x')
            
            else:
                recto=mpl.patches.Rectangle((xco,yco),wo, ho, color='black', fill=False)
                recti=mpl.patches.Rectangle((xci,yci),wi, hi, color='black', fill=False)
                ax.add_patch(recto)
                ax.add_patch(recti)
                ax.scatter(xco, yco, color='black', marker='x')

    return fig, ax

def plot3D(plotObject, ax, fine, cmap,
            norm, foc1, foc2, units=Units.MM, plotSystem_f=False):
    """!
    Plot a 3D reflector.

    @param plotObject A reflDict containing surface on which to plot beam. 
    @param ax Axis to use for plotting.
    @param fine Spacing of normals for plotting.
    @param cmap Colormap of reflector.
    @param norm Plot reflector normals.
    @param foc1 Plot focus 1.
    @param foc2 Plot focus 2.
    @param units Units to plot in.
    @param plotSystem_f Whether or not plot3D is called from plotSystem.
    """
    # Check that we haven't been asked for angular units, which won't work
    if units.dimension != 'spatial':
        units = Units.MM

    skip = slice(None,None,fine)
    grids = BRefl.generateGrid(plotObject, transform=True, spheric=True)

    ax.plot_surface(grids.x[skip]/units, grids.y[skip]/units, grids.z[skip]/units,
                   linewidth=0, antialiased=False, alpha=1, cmap=cmap)

    if foc1:
        try:
            ax.scatter(plotObject["focus_1"][0]/units, plotObject["focus_1"][1]/units, plotObject["focus_1"][2]/units, color='black')
        except KeyError as err:
            print_tb(err.__traceback__)


    if foc2:
        try:
            ax.scatter(plotObject["focus_2"][0]/units, plotObject["focus_2"][1]/units, plotObject["focus_2"][2]/units, color='black')
        except KeyError as err:
            print_tb(err.__traceback__)

    if norm:
        length = 10# np.sqrt(np.dot(plotObject["focus_1"], plotObject["focus_1"])) / 5
        skipn = slice(None,None,10*fine)
        ax.quiver(grids.x[skipn,skipn]/units, grids.y[skipn,skipn]/units, grids.z[skipn,skipn]/units,
                        grids.nx[skipn,skipn]/units, grids.ny[skipn,skipn]/units, grids.nz[skipn,skipn]/units,
                        color='black', length=length/units, normalize=True)

    if not plotSystem_f:
        ax.set_ylabel(f"$y$ ({units.name})", labelpad=20)
        ax.set_xlabel(f"$x$ ({units.name})", labelpad=10)
        ax.set_zlabel(f"$z$ ({units.name})", labelpad=50)
        ax.set_title(plotObject["name"], fontsize=20)
        world_limits = ax.get_w_lims()
        ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
        ax.tick_params(axis='x', which='major', pad=-3)
        ax.minorticks_off()

    del grids

def plotSystem(systemDict, ax, fine, cmap,norm,
            foc1, foc2, RTframes, RTcolor, units=Units.MM, title=None):
    """!
    Plot the system.

    @param systemDict Dictionary containing the reflectors to be plotted.
    @param ax Axis of plot.
    @param fine Spacing of normals for plotting.
    @param cmap Colormap of reflector.
    @param norm Plot reflector normals.
    @param foc1 Plot focus 1.
    @param foc2 Plot focus 2.
    @param RTframes List containing frames to be plotted.
    @param units Units to plot system in.
    """
    # Check that we've been asked to plot things in spatial units
    if units.dimension != 'spatial':
        units = Units.MM

    for i, (key, refl) in enumerate(systemDict.items()):
        if isinstance(cmap, list):
            _cmap = cmap[i]

        else:
            _cmap = cmap

        plot3D(refl, ax, fine=fine, cmap=_cmap,
                    norm=norm, foc1=foc1, foc2=foc2, units=units, plotSystem_f=True)
    
    ax.set_ylabel(f"$y$ ({units.name})", labelpad=20)
    ax.set_xlabel(f"$x$ ({units.name})", labelpad=10)
    ax.set_zlabel(f"$z$ ({units.name})", labelpad=20)
    if title is not None:
        ax.set_title(title, fontsize=20)
    world_limits = ax.get_w_lims()

    ax.tick_params(axis='x', which='major', pad=-3)

    if RTframes:
        for i in range(RTframes[0].size):
            x = []
            y = []
            z = []

            for frame in RTframes:
                x.append(frame.x[i])
                y.append(frame.y[i])
                z.append(frame.z[i])

            ax.plot(x/units, y/units, z/units, color=RTcolor, zorder=100, lw=0.7)


    #set_axes_equal(ax)
    ax.minorticks_off()
    #ax.set_box_aspect((1,1,1))
    ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

def plotBeamCut(x_cut, y_cut, x_strip, y_strip, vmin, vmax, units, scale=Scales.dB, figax=None, labels=None):
    """!
    Plot two beam cuts in the same figure.

    @param x_cut E-plane.
    @param y_cut H-plane.
    @param x_strip Co-ordinates for plotting E-plane.
    @param y_strip Co-ordinates for plotting H-plane.
    @param vmin Minimum for plot range.
    @param vmax Maximum for plot range.
    @param units Unit for x-axis. Instance of Units enum object.

    @returns fig Plot figure.
    @returns ax Plot axis.
    """
    if figax is None:
        fig, ax = pt.subplots(1,1, figsize=(5,5)) 
    else:
        fig, ax = figax

    if labels is None:
        labels = ["E-plane", "H-plane"]

    ax.plot(x_strip/units, x_cut, label=labels[0])
    ax.plot(y_strip/units, y_cut, ls="dashed", label=labels[1])

    ax.set_xlim(np.nanmin(x_strip/units), np.nanmax(x_strip/units))
    ax.set_ylim(vmin, vmax)

    ax.set_xlabel(f"$\\theta$ ({units.name})")
    if scale.name is 'dB':
        ax.set_ylabel("Power (dB)")
    elif scale.name is 'LIN':
        ax.set_ylabel("Power (Watts)")
    else:
        ax.set_ylabel("Amplitude (√W)")
    ax.legend(frameon=False, prop={'size': 13},handlelength=1)

    return fig, ax

def plotRTframe(frame, project, savePath, returns, aspect, units):
    """!
    Plot a ray-trace frame spot diagram.

    @param frame A PyPO frame object.
    @param project Set abscissa and ordinate of plot. Should be given as an instance of the Projection enum.
    @param savePath Path to save plot to.
    @param returns Whether to return figure object.
    @param aspect Aspect ratio of plot.
    @param units Units of the axes for the plot. Instance of Units enum object.
    """

    fig, ax = pt.subplots(1,1, figsize=(5,5))

    idx_good = np.argwhere((frame.dx**2 + frame.dy**2 + frame.dz**2) > 0.8)

    if project == Projections.xy:
        ax.scatter(frame.x[idx_good]/units, frame.y[idx_good]/units, color="black", s=10)
        ax.set_xlabel(f"$x$ ({units.name})")
        ax.set_ylabel(f"$y$ ({units.name})")

    elif project == Projections.xz:
        ax.scatter(frame.x[idx_good]/units, frame.z[idx_good]/units, color="black", s=10)
        ax.set_xlabel(f"$x$ ({units.name})")
        ax.set_ylabel(f"$z$ ({units.name})")
    
    elif project == Projections.yz:
        ax.scatter(frame.y[idx_good]/units, frame.z[idx_good]/units, color="black", s=10)
        ax.set_xlabel(f"$y$ ({units.name})")
        ax.set_ylabel(f"$z$ ({units.name})")
    
    elif project == Projections.yx:
        ax.scatter(frame.y[idx_good]/units, frame.x[idx_good]/units, color="black", s=10)
        ax.set_xlabel(f"$y$ ({units.name})")
        ax.set_ylabel(f"$x$ ({units.name})")

    elif project == Projections.zy:
        ax.scatter(frame.z[idx_good]/units, frame.y[idx_good]/units, color="black", s=10)
        ax.set_xlabel(f"$z$ ({units.name})")
        ax.set_ylabel(f"$y$ ({units.name})")
    
    elif project == Projections.zx:
        ax.scatter(frame.z[idx_good]/units, frame.x[idx_good]/units, color="black", s=10)
        ax.set_xlabel(f"$z$ ({units.name})")
        ax.set_ylabel(f"$x$ ({units.name})")

    ax.set_aspect(aspect)
    
    if returns:
        return fig

    pt.show()
