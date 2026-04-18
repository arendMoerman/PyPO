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
from PyPO.Enums import Projections, GridModes, FieldComponents, CurrentComponents, Units, Scales

def plotBeam2D(reflGrid, field, gmode=GridModes.xy,
               contour=None, contour_comp=FieldComponents.Ex, levels=None, 
               aperDict=None,
               norm=True, vmin=None, vmax=None, scale=Scales.dB, 
               amp_only=False, unwrap_phase=False, correct_phase=False, k=None,
               project=Projections.xy, units=Units.MM, 
               figax=None, figsize=None, 
               title=None, titleA="Amplitude", titleP="Phase", **kwargs):
    """!
    Generate a 2D plot of a field or current.

    @param reflGrid A reflGrid containing the surface grid which to plot beam. 
    @param field PyPO field or current component to plot.
    @param contour A fields or currents object to plot as contour.
    @param contour_comp The component of the contour fields or currents object to plot as contours
    @param levels Levels for contourplot.
    @param aperDict Plot an aperture defined in an aperDict object along with the field or current patterns. Default is None.
    @param norm Normalise field.
    @param vmin Minimum amplitude value to display. Default is -30.
    @param vmax Maximum amplitude value to display. Default is 0.
    @param scale Plot amplitude in decibels, logarithmic or linear scale. Instance of Scales enum object.
    @param amp_only Only plot amplitude pattern. Default is False.
    @param unwrap_phase Unwrap the phase pattern. Prevents annular structure in phase pattern. Default is False.
    @param correct_phase Boolean or 3 element numpy array. Applies a phase factor to the field equal to
            k*displacement of the grid along the Z-axis (True) or direction of the 3-vector.
    @param k Wavenumber to use for phase correction. Only used if correct_phase is not False.
    @param project Set abscissa and ordinate of plot. Should be given as an instance of the Projection enum.
    @param units The units of the axes. Instance of Units enum object.
    @param figax Tuple of Matplotlib Figure and Axes object or np.array of Axes objects to use for the plots. If none, create new figure.
    @param figsize Tuple with Matplotlib Figure size. Defaults to (10,5) or (5,5). Has no effect if figax is given.
    @param title An overall title for the plot. Defaults to the field name and component.
    @param titleA Title of the amplitude plot. Default is "Amp".
    @param titleP Title of the phase plot. Default is "Phase".
    @param **kwargs Additional keywords to pass to Axes.pcolormesh Matplotlib methods.
    
    @returns fig Figure object containing plot.
    @returns ax Axes containing the axes of the plot.

    @see aperDict
    """
    if isinstance(correct_phase, bool) and correct_phase:
        correct_phase = 1

    if isinstance(gmode, str):
        gmode = GridModes._member_map_[gmode]
    elif isinstance(gmode, int):
        gmode = GridModes._value2member_map_[gmode]

    if not gmode == GridModes.AoE:
        if units.dimension != 'spatial':
            units = Units.MM
        if project == Projections.xy:
            grid_x1 = reflGrid.x
            grid_x2 = reflGrid.y
            ff_flag = False
            comps = ["x", "y"]
            if isinstance(correct_phase, int):
                if correct_phase: 
                    correct_phase = np.array((0,0, np.sign(correct_phase)))

        elif project == Projections.yz:
            grid_x1 = reflGrid.y
            grid_x2 = reflGrid.z
            ff_flag = False
            comps = ["y", "z"]
            if isinstance(correct_phase, int):
                if correct_phase: 
                    correct_phase = np.array((np.sign(correct_phase, 0, 0)))

        elif project == Projections.zx:
            grid_x1 = reflGrid.z
            grid_x2 = reflGrid.x
            ff_flag = False
            comps = ["z", "x"]
            if isinstance(correct_phase, int):
                if correct_phase: 
                    correct_phase = np.array((0, np.sign(correct_phase)), 0)

        elif project == Projections.yx:
            grid_x1 = reflGrid.y
            grid_x2 = reflGrid.x
            ff_flag = False
            comps = ["y", "x"]
            if isinstance(correct_phase, int):
                if correct_phase: 
                    correct_phase = np.array((0,0, np.sign(correct_phase)))

        elif project == Projections.zy:
            grid_x1 = reflGrid.z
            grid_x2 = reflGrid.y
            ff_flag = False
            comps = ["z", "y"]
            if isinstance(correct_phase, int):
                if correct_phase: 
                    correct_phase = np.array((np.sign(correct_phase, 0, 0)))

        elif project == Projections.xz:
            grid_x1 = reflGrid.x
            grid_x2 = reflGrid.z
            ff_flag = False
            comps = ["x", "z"]
            if isinstance(correct_phase, int):
                if correct_phase: 
                    correct_phase = np.array((0, np.sign(correct_phase)), 0)

    else: # a farfield grid
        if units.dimension != 'angular':
            units = Units.DEG
            
        if project == Projections.xy:
            grid_x1 = reflGrid.x
            grid_x2 = reflGrid.y
            ff_flag = True
            comps = ["\\mathrm{Az}", "\\mathrm{El}"]

        elif project == Projections.yx:
            grid_x1 = reflGrid.y
            grid_x2 = reflGrid.x
            ff_flag = True
            comps = ["\\mathrm{El}", "\\mathrm{Az}"]
            
        else:
            raise ValueError("Cannot form projections involving z for farfield grids")

        if not ((units == Units.DEG) or (units == Units.AM) or (units == Units.AS)):
            units = Units.DEG
            
    if not amp_only:
        # Set a default figure size if none are given
        if figsize is None:
            figsize = (10,5)
            
        if not figax:
            fig, ax = pt.subplots(1,2, figsize=figsize, gridspec_kw={'wspace':0.5})
        else:
            fig, ax = figax

        if correct_phase is not False:
            if isinstance(correct_phase, int) or isinstance(correct_phase, float):
                correct_phase = int(np.sign(correct_phase))
                # Correct phase for z-axis displacement
                if gmode == GridModes.uv:
                    correct_phase = correct_phase*np.array((np.mean(reflGrid.nx[0,:]), np.mean(reflGrid.ny[0,:]), np.mean(reflGrid.nz[0,:])))
                    vnorm = correct_phase / np.linalg.vector_norm(correct_phase)
                else:
                    shape = np.array(reflGrid.z.shape)
                    n = int(shape[0]/2)
                    m = int(shape[1]/2)
                    vnorm = correct_phase*np.array((reflGrid.nx[n,m], reflGrid.ny[n,m], reflGrid.nz[n, m]))
            else: # Correct_phase is a vector
                try:
                    if len(correct_phase) != 3:
                        raise ValueError
                except (ValueError, KeyError, TypeError):
                    raise ValueError("correct_phase must be either boolean, number or np.ndarray((nx, ny, nz))")
                vnorm = correct_phase / np.linalg.vector_norm(correct_phase)
            
            offset = np.linalg.vecdot(vnorm, np.stack((reflGrid.x, reflGrid.y, reflGrid.z), axis=-1))

            if gmode == GridModes.uv:
                r0 = np.mean(offset[0,:])
                phase_factor = np.exp(-1j*k*(offset-r0))
            elif gmode == GridModes.xy:
                r0 = offset[n,m]
                phase_factor = np.exp(-1j*k*(offset-r0))
            else: # Don't know what to do for farfields
                phase_factor = 1.0
        else:
            phase_factor = 1.0
        
        if scale == Scales.LIN:
            if norm:
                max_field = np.nanmax(np.absolute(field))
                field_pl = np.absolute(field) / max_field
                if contour is not None:
                    contour = np.absolute(contour) / np.nanmax(np.absolute(contour))
            
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
                                    vmin=vmin, vmax=vmax, **kwargs)
            phasefig = ax[1].pcolormesh(grid_x1/units, grid_x2/units, phase, **kwargs)

            if contour is not None:
                cont0 = ax[0].contour(grid_x1/units, grid_x2/units, contour**2, levels, cmap=cm.binary, linewidths=0.5)
                cont1 = ax[1].contour(grid_x1/units, grid_x2/units, np.angle(contour), levels, cmap=cm.binary, linewidths=0.5)

                ax[0].clabel(cont0)
                ax[1].clabel(cont1)

        if scale == Scales.AMP:
            if norm:
                max_field = np.nanmax(np.absolute(field))
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
                                    vmin=vmin, vmax=vmax, **kwargs)
            phasefig = ax[1].pcolormesh(grid_x1 / units.value, grid_x2 / units.value, phase, **kwargs)

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
                max_field = np.nanmax(np.absolute(field))
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
                                    vmin=vmin, vmax=vmax, **kwargs)
            phasefig = ax[1].pcolormesh(grid_x1/units, grid_x2/units, phase, **kwargs)
            

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

    # Amplitude only plotting
    else:
        if figsize is None:
            figsize = (5,5)
          
        if not figax:
            fig, ax = pt.subplots(1,1, figsize=figsize, gridspec_kw={'wspace':0.5})
        else:
            fig, ax = figax
            
        if isinstance(ax, np.ndarray):
            ax = ax[0]

        divider = make_axes_locatable(ax)
        # TODO: Need to fix colorbar height and location.  Gridspec does not work for single plot
        cax = divider.append_axes('right', size='5%', pad=0.05)

        if scale == Scales.LIN:
            if norm:
                max_field = np.nanmax(np.absolute(field))
                field_pl = np.absolute(field) / max_field
                if contour is not None:
                    contour_pl = np.absolute(contour) / np.max(np.absolute(contour))
            
            else:
                field_pl = np.absolute(field)
                if contour is not None:
                    contour_pl = np.absolute(contour)

            vmin = np.min(field_pl)**2 if vmin is None else vmin
            vmax = np.max(field_pl)**2 if vmax is None else vmax
            
            ampfig = ax.pcolormesh(grid_x1/units, grid_x2/units, field_pl**2,
                                    vmin=vmin, vmax=vmax, **kwargs)

            if contour is not None:
                cont = ax.contour(grid_x1/units, grid_x2/units, contour_pl**2, levels, cmap=cm.binary, linewidths=0.5)
                ax.clabel(cont)
                
        elif scale==Scales.AMP:
            if titleA == "Amplitude":
                titleA += " / √W"
            
            if norm:
                max_field = np.nanmax(np.absolute(field))
                field_pl = np.absolute(field) / max_field
            else:
                field_pl = np.absolute(field)
                
            if contour is not None:
                if norm:
                    contour_pl = np.absolute(contour) / np.nanmax(np.absolute(contour))
                else:
                    contour_pl = np.absolute(contour)
            
            vmin = np.min(field_pl) if vmin is None else vmin
            vmax = np.max(field_pl) if vmax is None else vmax
            
            ampfig = ax.pcolormesh(grid_x1/units, grid_x2/units, field_pl,
                                    vmin=vmin, vmax=vmax, **kwargs)
            
            if contour is not None:
                cont = ax.contour(units.rdiv(grid_x1), units.rdiv(grid_x2), contour_pl, levels, cmap=cm.binary, linewidths=0.5)
                ax.clabel(cont)
        else: # scale == Scales.dB:
            if titleA == "Power":
                titleA += " / dB"
            
            if norm:
                max_field = np.nanmax(np.absolute(field))
                field_dB = 20 * np.log10(np.absolute(field) / max_field)
            else:
                field_dB = 20 * np.log10(np.absolute(field))
                
            if contour is not None:
                if norm:
                    contour_dB = 20 * np.log10(np.absolute(contour) / np.max(np.absolute(contour)))
                else:
                    contour_dB = 20 * np.log10(np.absolute(contour))
            
            vmin = np.min(field_dB) if vmin is None else vmin
            vmax = np.max(field_dB) if vmax is None else vmax
            
            ampfig = ax.pcolormesh(grid_x1/units, grid_x2/units, field_dB,
                                    vmin=vmin, vmax=vmax, **kwargs)
            
            if contour is not None:
                cont = ax.contour(units.rdiv(grid_x1), units.rdiv(grid_x2), contour_dB, levels, cmap=cm.binary, linewidths=0.5)
                ax.clabel(cont)

        ax.set_ylabel(r"${}$ ({})".format(comps[1], units.name))
        ax.set_xlabel(r"${}$ ({})".format(comps[0], units.name))

        ax.set_title(titleA, y=1.02)
        ax.set_box_aspect(1)

        c = fig.colorbar(ampfig, cax=cax, orientation='vertical')
    
    if title is not None:
        fig.suptitle(title)  # Set a position here
    
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

            ax.plot(np.array(x)/units, np.array(y)/units, np.array(z)/units, color=RTcolor, zorder=100, lw=0.7)


    #set_axes_equal(ax)
    ax.minorticks_off()
    #ax.set_box_aspect((1,1,1))
    ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

def plotBeamCut(strip, cut, units=Units.DEG, vmin=None, vmax=None, scale=Scales.dB, 
                norm=False, amp_only=True, figax=None, title=None, **kwargs):
    """!
    Plot a beam cut

    @param strip Coordinates for plotting the cut.
    @param cut Field values to plot
    @param units Unit for x-axis. Instance of Units enum object.
    @param vmin Minimum for plot range.
    @param vmax Maximum for plot range.
    @param scale The units of the y axis. Instance of Scales Enum object.
    @param norm Normalise field.
    @param amp_only Only plot amplitude pattern. Default is True.
    @param figax Tuple of Matplotlib Figure and Axes object or np.array of Axes objects to use for the plots. If none, create new figure.
    @param title An overall title for the plot. Defaults to the field name and component.
    @param **kwargs Additional keyword arguments to pass to the Matplotlib Axes.plot method

    @returns fig Plot figure.
    @returns ax Plot axis.
    """
    if figax is None:
        if not amp_only:
            fig, ax = pt.subplots(1,2, figsize=(10,5))
        else:
            fig, ax = pt.subplots(1,1, figsize=(5,5)) 
    else:
        fig, ax = figax

    if scale.name == 'dB':
        ylabel = "Power (dB)"
        field = 20*np.log10(np.abs(cut))
    elif scale.name == 'LIN':
        ylabel = "Power (Watts)"
        field = np.abs(cut)**2
    else:
        ylabel = "Amplitude (√W)"
        field = np.abs(cut)
        
    if norm:
        fnorm = np.nanmax(field)
        ylabel = "Normalized " + ylabel
    else:
        fnorm = 1.0
        
    if vmax is None:
        vmax = np.nanmax(field/fnorm)
    
    if vmin is None:
        vmin = np.nanmin(field/fnorm)
        
    if units.dimension == 'angular':
        xlabel = f'$\\theta$ ({units.name})'
    else:
        xlabel = f'$\\rho$ ({units.name})'
    
    if not amp_only:
        ax[0].plot(strip/units, field/fnorm, **kwargs)
        ax[1].plot(strip/units, np.angle(cut), **kwargs)
        
        if figax is None:
            ax[0].set_ylim(vmin, vmax)

        ax[0].set_xlabel(xlabel)
        ax[1].set_xlabel(xlabel)
        
        ax[0].set_xlabel(ylabel)
        ax[1].set_xlabel("Phase (rad)")
        
        ax.legend(frameon=False, prop={'size': 13},handlelength=1)

    else:
        ax.plot(strip/units, field/fnorm, **kwargs)
        
        if figax is None:
            ax.set_ylim(vmin, vmax)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        ax.legend(frameon=False, prop={'size': 13},handlelength=1)
        
    if title:
        fig.suptitle(title)

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
