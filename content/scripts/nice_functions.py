import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

import ipywidgets as widgets
from ipywidgets import AppLayout, FloatSlider
from ipywidgets import GridspecLayout
from matplotlib import rc

from copy import deepcopy

try:
    from addict import Dict 
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "addict"])
    from addict import Dict 

try:
    import ipympl
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ipympl"])
    
    
def dict_merge(destination, source):
    out = deepcopy(destination)
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = out.setdefault(key, {})
            dict_merge(node, value)
        else:
            out[key] = value
    return out

PARAM = Dict()
PARAM.background_color = (0.0, 0.0, 0.0, 0.0)  
PARAM.zlim = (-4,4)
PARAM.grid_color = (0.0, 0.0, 0.0, 0.25)
PARAM.grid_linewidth = 0.25
PARAM.grid_size = 20
PARAM.border_color = (0.0, 0.0, 0.0, 0.25)
PARAM.border_linewidth = 0.25
PARAM.tick_color = (0.0, 0.0, 0.0, 0.25)
PARAM.tick_linewidth = 0.25
PARAM.label_size = 7
PARAM.label_color = (0,0,0,1)
PARAM.azimuth = -60
PARAM.elevation = 30
PARAM.header_visible = False
PARAM.style = ""
PARAM.plot_box = [[-2,2],[-2,2]]
PARAM.cmap = plt.cm.jet
PARAM.levels_number = 15
PARAM.density = 0.5
PARAM.iter_max = 40
    

def set_axis3d(ax=None, **param):
    # api for 3D plot : https://matplotlib.org/3.1.1/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html
    # last component is a level of gray : 0.0 for white to 1.0 for black
    # https://matplotlib.org/_modules/mpl_toolkits/mplot3d/axes3d.html
    
    param = Dict(dict_merge(PARAM, param))
    if ax is None:
        ax = plt.axes(projection='3d')
    
    if param.style == "raw":
        param.grid_color = (0.0, 0.0, 0.0, 0)
        param.border_color = (0.0, 0.0, 0.0, 0)
        param.tick_color = (0.0, 0.0, 0.0, 0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    # Now we apply the parameters:
    
    # Background of the plot
    ax.w_xaxis.set_pane_color(param.background_color)
    ax.w_yaxis.set_pane_color(param.background_color)
    ax.w_zaxis.set_pane_color(param.background_color)

    # Grid of the background
    ax.xaxis._axinfo["grid"]['color'] = param.grid_color
    ax.yaxis._axinfo["grid"]['color'] = param.grid_color
    ax.zaxis._axinfo["grid"]['color'] = param.grid_color
    ax.xaxis._axinfo["grid"]['linewidth'] = param.grid_linewidth
    ax.yaxis._axinfo["grid"]['linewidth'] = param.grid_linewidth
    ax.zaxis._axinfo["grid"]['linewidth'] = param.grid_linewidth

    # Main axes (with the ticks labels etc) : 
    ax.w_xaxis.line.set_color(param.border_color) 
    ax.w_yaxis.line.set_color(param.border_color) 
    ax.w_zaxis.line.set_color(param.border_color)
    ax.w_xaxis.line.set_linewidth(param.border_linewidth) 
    ax.w_yaxis.line.set_linewidth(param.border_linewidth) 
    ax.w_zaxis.line.set_linewidth(param.border_linewidth) 

    # Ticks
    ax.tick_params(color=param.tick_color, 
                   size=param.tick_linewidth, # this is broken..
                   axis='both', 
                   which='both', 
                   labelsize=param.label_size, 
                   labelcolor=param.label_color)
    # other options : ['size', 'width', 'color', 'tickdir', 'pad', 'labelsize', 'labelcolor', 'zorder', 'gridOn', 'tick1On', 'tick2On', 'label1On', 'label2On', 'length', 'direction', 'left', 'bottom', 'right', 'top', 'labelleft', 'labelbottom', 'labelright', 'labeltop', 'labelrotation', 'grid_agg_filter', 'grid_alpha', 'grid_animated', 'grid_antialiased', 'grid_clip_box', 'grid_clip_on', 'grid_clip_path', 'grid_color', 'grid_contains', 'grid_dash_capstyle', 'grid_dash_joinstyle', 'grid_dashes', 'grid_data', 'grid_drawstyle', 'grid_figure', 'grid_fillstyle', 'grid_gid', 'grid_in_layout', 'grid_label', 'grid_linestyle', 'grid_linewidth', 'grid_marker', 'grid_markeredgecolor', 'grid_markeredgewidth', 'grid_markerfacecolor', 'grid_markerfacecoloralt', 'grid_markersize', 'grid_markevery', 'grid_path_effects', 'grid_picker', 'grid_pickradius', 'grid_rasterized', 'grid_sketch_params', 'grid_snap', 'grid_solid_capstyle', 'grid_solid_joinstyle', 'grid_transform', 'grid_url', 'grid_visible', 'grid_xdata', 'grid_ydata', 'grid_zorder', 'grid_aa', 'grid_c', 'grid_ds', 'grid_ls', 'grid_lw', 'grid_mec', 'grid_mew', 'grid_mfc', 'grid_mfcalt', 'grid_ms']
    
    # title and stuff around the plot
    if "title" in param.keys():    
        ax.title.set_text(param["title"])
    ax.view_init(param.elevation, param.azimuth)
    
    return ax
    

def function_values(fun, **param):
    # fun is a function taking two floats as an argument, returning one
    if "plot_box" not in param.keys():
        param["plot_box"] = [[-2,2],[-2,2]]
    if "grid_size" not in param.keys():
        param["grid_size"] = 20
    
    x = np.outer(np.linspace(param["plot_box"][0][0], param["plot_box"][0][1], param["grid_size"]), np.ones(param["grid_size"]))
    y = np.outer(np.linspace(param["plot_box"][1][0], param["plot_box"][1][1], param["grid_size"]), np.ones(param["grid_size"])).T
    
    z = np.zeros((param["grid_size"],param["grid_size"]))
    for idx, _ in np.ndenumerate(x):
        z[idx] = fun(x[idx], y[idx])
    return x, y, z


def field_values(fun, **param):
    # fun is a function taking two floats as an argument, returning two
    if "plot_box" not in param.keys():
        param["plot_box"] = [[-2,2],[-2,2]]
    if "grid_size" not in param.keys():
        param["grid_size"] = 20
    
    x = np.outer(np.linspace(param["plot_box"][0][0], param["plot_box"][0][1], param["grid_size"]), np.ones(param["grid_size"]))
    y = np.outer(np.linspace(param["plot_box"][1][0], param["plot_box"][1][1], param["grid_size"]), np.ones(param["grid_size"])).T
    
    zx = np.zeros((param["grid_size"],param["grid_size"]))
    zy = np.zeros((param["grid_size"],param["grid_size"]))
    for idx, _ in np.ndenumerate(x):
        zx[idx], zy[idx] = fun(x[idx], y[idx])
    return x, y, zx, zy


def matrix_definiteness(A):
    if np.linalg.norm(A-A.T)> 1e-8 : # not symmetric
        return matrix_definiteness(0.5*(A+A.T))
    spec = np.linalg.eig(A)[0]
    if (spec>0).all():
        return "Définie Positive"
    elif (spec>=0).all():
        return "Positive"
    elif (spec<0).all():
        return "Définie Négative"
    elif (spec<=0).all():
        return "Négative"
    else:
        return "Non Définie"

def matrix_symmetricness(A):
    if np.linalg.norm(A-A.T) < 1e-8:
        return "symétrique"
    elif np.linalg.norm(A+A.T) < 1e-8:
        return "antisymétrique"
    else:
        return "non symétrique"

def print_info(A):
    return "La matrice A est " + matrix_definiteness(A) + ", " +  matrix_symmetricness(A)


def get_plots_we_want(**plot_param):
    # what do we want to plot? Let's look at the parameters
    possible_plots = ["graph", "levelset", "gradient", "flow"]
    have_to_plot = {plot : ((plot in plot_param.keys()) and plot_param[plot]) for plot in possible_plots}# a dict with bool values
    number_have_to_plot = sum(list(have_to_plot.values())) # number of Trues in the dict
    
    return have_to_plot, number_have_to_plot


def is_in_box(x, box):
    return box[0][0] < x[0] < box[0][1] and box[1][0] < x[1] < box[1][1]


def sequence_gradient(func, x0=np.ones(2),stepsize=0.1,iter_max=100, **param):
    # here func is a function with a gradient parameter
    seq = []
    x = x0
    if 'plot_box' in param.keys() and is_in_box(x, param['plot_box']):
        seq.append(x)
    for t in range(iter_max):
        grad = func(x[0], x[1], gradient=True) # grad is a tuple, we can compine it with arrays
        x = x - stepsize*np.array(grad)
        if 'plot_box' in param.keys():
            # we check that the sequence is still within the bounds of our plot.
            if is_in_box(x, param['plot_box']):
                seq.append(x)
        else:
            seq.append(x)
    X = [x[0] for x in seq]
    Y = [x[1] for x in seq]
    return X, Y

def sequence_newton(func, x0=np.ones(2), stepsize=1, iter_max=10, **param):
    # here func is a function with a gradient and hessian parameter
    seq = []
    x = x0
    if 'plot_box' in param.keys() and is_in_box(x, param['plot_box']):
        seq.append(x)
    for t in range(iter_max):
        grad = func(x[0], x[1], gradient=True) # grad is a tuple, we can compine it with arrays
        hess = func(x[0], x[1], hessian=True) # hess is a 2D array
        d = np.linalg.solve(hess, grad)
        x = x - stepsize*d
        if 'plot_box' in param.keys():
            # we check that the sequence is still within the bounds of our plot.
            if is_in_box(x, param['plot_box']):
                seq.append(x)
        else:
            seq.append(x)
    X = [x[0] for x in seq]
    Y = [x[1] for x in seq]
    return X, Y



def quadratic(A, b=np.zeros(2), c=0):
    # returns the function 0.5*<AX,X> + <b,X> + c or its gradient
    def func(x,y, gradient=None, hessian=None):
        if gradient is None and hessian is None:
            return 0.5*( x*(A[0,0]*x + A[0,1]*y) + y*(A[1,0]*x + A[1,1]*y) ) + b[0]*x + b[1]*y + c
        elif gradient:
            AA = 0.5*(A + A.T)
            return ( AA[0,0]*x + AA[0,1]*y + b[0], AA[1,0]*x + AA[1,1]*y + b[1] )
        elif hessian:
            return 0.5*(A + A.T)
    return func
































def plot2d_function(func, fig=None, **plot_param):
    """ Reprensents a func : (x,y) ---> z in various ways
        If we ask for more than one representation, 
        they are all displayed next to each other in a subplot
        
    INPUT:
      - fig : the handle of the figure in which drawing
      - func : a function with the signature func(x,y,gradient) 
            func(x,y) returns the value of the function at (x,y)
            func(x,y,gradient=True) returns the value of the 
            gradient at (x,y), as a 2-tuple
        
    OUTPUT: None
        
    OPTIONAL PARAMETERS: there is a *lot* of them, here are the main ones. default values
        are stored in the PARAM dictionary
      - cmap, colormap (plt.cm.jet) : the same colormap is used for every plot
      - style, str (None) : "raw" plots naked graphs, without axis, background or ticks
      - plot_box, 2D array ([[-2,2],[-2,2]]) : The syntax is [[xmin, xmax],[ymin,ymax]].
            Defines the box in which all the objects will be plotted
      
      - graph, bool (False) : tells if plotting the 3D graph of the function. Displayed by default.
          - grid_size, int (20) : Controls the precision of the mesh for plotting the graph.
          - title, str (None) : A title to display on top of the graph
      
      - levelset, bool (False) : tells if plotting the 2D levelsets of the function
          - levels, int (15) : number of levelsets to be displayed
          
      - gradient, bool (False) : tells if plotting the 2D vector field of the gradient of the function
      
      - flow, bool (False) : tells if plotting the descent gradient flow curves of the function
          - density, float (0.5) : density of the flow curves. Impacts the performance.
          
      - sequence, (list,list) (None) : if a sequence is specified, it will be plotted on top of the 2D
            graphs (levelset, gradient, flow). Impossible to plot it on 3D surfaces so far. 
            The syntax is : a 2-tuple, each component being a list of floats (same size) representing
            the x- and y-coordinates of the sequence.
      - algo, str (None) : instead of specifying a sequence, you can just pass an argument to tell
            which algorithm you want to run on the function. Currently supported:
          - "gradient" : Runs the gradient descent algorithm. You can pass options:
              - x0, 1D arrray (np.ones(2)) : the initialization of the algorithm
              - iter_max, int (100) : the maximum number of iterations allowed
              - stepsize, float (0.1) : the stepsize in the iteration x <-- x - stepsize*gradient
    """
    # deal with optional parameters
    plot_param = Dict(dict_merge(PARAM, plot_param))
    have_to_plot, number_plots = get_plots_we_want(**plot_param)
    # untangle 2D/1D inconsistencies
    func = flatten_2D_function(func)
    
    if fig is None:
        fig = plt.gcf()
        
    if 'algo' in plot_param.keys() and plot_param.algo is not None:
        if plot_param.algo == 'gradient':
            seq = sequence_gradient(func, **plot_param)
            if len(seq[0]) == 0: # we are out of bounds
                plot_param.sequence = None
            else:
                plot_param.sequence = seq
        if plot_param.algo == 'Newton':
            seq = sequence_newton(func, **plot_param)
            if len(seq[0]) == 0: # we are out of bounds
                plot_param.sequence = None
            else:
                plot_param.sequence = seq
    if 'sequence' in plot_param.keys() and plot_param['sequence'] is not None:
        s = np.ones(len(plot_param.sequence[0]))*40
        s[0] = 120
        plot_param.scatter_size = s
                
    fig.canvas.header_visible = plot_param.header_visible
    
    k = 0
    if have_to_plot['graph']:
        k = k+1
        ax = fig.add_subplot(1, number_plots, k, projection='3d')
        ax = set_axis3d(ax, **plot_param)
        ax.plot_surface(*function_values(func, **plot_param), cmap=plot_param.cmap, rstride=1, cstride=1);
        ax.set_zlim(plot_param.zlim);
        
    if have_to_plot['levelset']:
        k = k+1
        ax = fig.add_subplot(1, number_plots, k)
        contour = ax.contour(*function_values(func, **plot_param), cmap=plot_param.cmap, levels=plot_param.levels_number)
        if plot_param["style"] != "raw":
            ax.clabel(contour, inline=1, fontsize=7);
        else:
            plt.axis('off')
        if 'sequence' in plot_param.keys() and plot_param['sequence'] is not None:
            ax.scatter(*plot_param['sequence'], s=plot_param.scatter_size)
            ax.plot(*plot_param['sequence'])
            
    if have_to_plot['gradient']:
        k = k+1
        ax = fig.add_subplot(1, number_plots, k)
        X,Y,U,V = field_values(lambda x,y : func(x,y,gradient=True), **plot_param)
        _,_,COLOR = function_values(func, **plot_param)
        contour = ax.quiver(X,Y,U,V,COLOR, cmap=plot_param.cmap)
        if plot_param["style"] == "raw":
            plt.axis('off')
        if 'sequence' in plot_param.keys() and plot_param['sequence'] is not None:
            ax.scatter(*plot_param['sequence'], s=plot_param.scatter_size)
            
    if have_to_plot['flow']:
        k = k+1
        ax = fig.add_subplot(1, number_plots, k)
        X,Y,U,V = field_values(lambda x,y : func(x,y,gradient=True), **plot_param)
        _,_,COLOR = function_values(func, **plot_param)
        # Here we need to TRANSPOSE the vector field because??? streamplot is stupid? arrays are indexed in a different way??
        contour = ax.streamplot(X.T[0],Y[0],-U.T,-V.T, color=COLOR, density=plot_param.density, cmap=plot_param.cmap)
        if plot_param["style"] == "raw":
            plt.axis('off')
        if 'sequence' in plot_param.keys() and plot_param['sequence'] is not None:
            ax.scatter(*plot_param['sequence'], s=plot_param.scatter_size)
    #print(plot_param)
    #return fig





def widget_quadratic(**plot_param):
    # given a function handle (taking two arguments, returning one)
    # plots an interactive widget with graph and level set
    
    # parameters for the plot
    default_param = {
        'plot_box' : [[-2,2],[-2,2]],
        'grid_size' : 30,
        'style' : "normal",
        'dpi' : 80,
        'sequence' : None,
        'algo' : None
    }
    if get_plots_we_want(**plot_param)[1] == 0:
        default_param['graph'] = True
    plot_param = { **default_param, **plot_param }
    dpi = plot_param['dpi']

    # define all the sliders we want to manipulate
    slider_param = {
        'min' : -2, 
        'max' : 2,
        'orientation' : 'horizontal',
        #'layout' : Layout(height='auto', width='auto')#Layout(width='40%', margin='0px 30% 0px 30%'),
    }
    slider_a11 = FloatSlider(description='$A_{11}$', value=2.0, **slider_param)
    slider_a12 = FloatSlider(description='$A_{12}$', value=0.0, **slider_param)
    slider_a21 = FloatSlider(description='$A_{21}$', value=0.0, **slider_param)
    slider_a22 = FloatSlider(description='$A_{22}$', value=1.0, **slider_param)
    slider_b1  = FloatSlider(description='$b_1$', value=0.0, **slider_param)
    slider_b2  = FloatSlider(description='$b_2$', value=0.0, **slider_param)
    if plot_param['algo'] is not None:
        slider_x0  = FloatSlider(description='$x_0$', value=1.0, **slider_param)
        slider_y0  = FloatSlider(description='$y_0$', value=1.0, **slider_param)
        slider_stepsize = FloatSlider(description='stepsize', value=0.25, min=0.01, max=2, step=0.01)
    
    # we initialize everything
    A_slider = np.array([[slider_a11.value, slider_a12.value], [slider_a21.value, slider_a22.value]])
    b_slider = np.array([slider_b1.value, slider_b2.value])
    plot_param["title"] = print_info(A_slider)
    if plot_param['algo'] is not None:
        x0_slider = np.array([slider_x0.value, slider_y0.value])
        plot_param['x0']= np.array([slider_x0.value, slider_y0.value])
        stepsize_slider = np.array([slider_stepsize.value])
        plot_param['stepsize'] = np.array([slider_stepsize.value])

    # open the figure
    plt.ioff() # turn off interactive mode to be able to display widget. I honestly don't understand why. see https://github.com/matplotlib/ipympl/issues/220 or https://github.com/matplotlib/matplotlib/pull/17371
    fig = plt.figure(dpi=dpi)
    plt.ion()
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    _, number_plots = get_plots_we_want(**plot_param)
    plt.gcf().set_size_inches(plt.figaspect(1/number_plots))
    # plot a first draw
    plot2d_function(quadratic(A_slider,b_slider), fig,  **plot_param)

    # the function handling the change of parameters
    # it seems that when stepsize/x0 is changed, the other parameter is reset
    # but there is no such problem with the coefficients of A..?
    # ok so more precisely : 
    def update_plot(change, slider, idx, plot_param):
        # store the camera 3D view, then clears the figure
        if 'graph' in plot_param.keys() and plot_param['graph']:
            plot_param['azimuth'] = fig.get_axes()[0].azim # remember there are two subplots
            plot_param['elevation'] = fig.get_axes()[0].elev
        fig.clear()
        # update the parameters
        A = A_slider
        b = b_slider
        if slider == 'A':
            A[idx] = change.new
        elif slider == 'b':
            b[idx] = change.new
        plot_param["title"] = print_info(A)
        if plot_param['algo'] is not None:
            plot_param['x0'] = x0_slider
            plot_param['stepsize'] = stepsize_slider[0]
            if slider == 'x0':
                plot_param['x0'][idx] = change.new
            if slider == 'stepsize':
                stepsize_slider[0] = change.new # why do we need to do that for this one specifically!?§ if not, when I change anything like A, x0, the stepsize is reset to the default value of the slider
                plot_param['stepsize'] = change.new
        # plot
        #plot_param["title"] = str(x0_slider) + str(stepsize_slider)
        plot2d_function(quadratic(A,b), fig, **plot_param)
        _=fig.canvas.draw()
        _=fig.canvas.flush_events()

    # keep track of changes
    slider_a11.observe(lambda change : update_plot(change, 'A', (0,0), plot_param), names='value')
    slider_a21.observe(lambda change : update_plot(change, 'A', (1,0), plot_param), names='value')
    slider_a12.observe(lambda change : update_plot(change, 'A', (0,1), plot_param), names='value')
    slider_a22.observe(lambda change : update_plot(change, 'A', (1,1), plot_param), names='value')
    slider_b1.observe(lambda change : update_plot(change, 'b', (0,), plot_param), names='value')
    slider_b2.observe(lambda change : update_plot(change, 'b', (1,), plot_param), names='value')
    if plot_param['algo'] is not None:
        slider_x0.observe(lambda change : update_plot(change, 'x0', (0,), plot_param), names='value')
        slider_y0.observe(lambda change : update_plot(change, 'x0', (1,), plot_param), names='value')
        slider_stepsize.observe(lambda change : update_plot(change, 'stepsize', None, plot_param), names='value')

    # now we display all of this
    # a grid of sliders for the parameters
    grid = GridspecLayout(2, 3)
    grid[0,0] = slider_a11
    grid[1,0] = slider_a21
    grid[0,1] = slider_a12
    grid[1,1] = slider_a22
    grid[0,2] = slider_b1
    grid[1,2] = slider_b2
    header=grid
    # optional sliders for the sequence
    if plot_param['algo'] is not None:
        grid2 = GridspecLayout(1, 3)
        grid2[0,0] = slider_x0
        grid2[0,1] = slider_y0
        grid2[0,2] = slider_stepsize
        footer=grid2
    else:
        footer=None
    
    # we gather everything
    return AppLayout(
        header=header,
        left_sidebar=None,
        center=fig.canvas,
        right_sidebar=None,
        footer=footer
    )



# what follows is terrible coding for dealing with 2D vectors versus 2x1D entries
# but it is the fastest way to do without having to rewrite everything

from inspect import signature
import autograd

def flatten_2D_function(func):
    # Given a function taking a vector entry z=(x,y) and returning f(z)
    # we return a function taking as first entries x and y to return f(x,y)
    nb_entry_param = len( signature(func).parameters )
    if nb_entry_param >= 2: # nothing to be done here
        return func
    else: # we flatten it
        def newfunc(x,y, gradient=None, hessian=None):
                x = float(x) # autograd doesn't like integers
                y = float(y)
                if gradient is None and hessian is None:
                    return func(np.array([x, y]))
                else:
                    if gradient:
                        dfz = autograd.grad(func)(np.array([x, y]))
                        return ( dfz[0], dfz[1] )
                    elif hessian:
                        ddfz = autograd.jacobian(autograd.grad(func))(np.array([x, y]))
                        return ddfz
        return newfunc   
    
    
def flatten_2D_sequence(list2D):
    if len(list2D) == 2:
        return list2D
    else:
        listx = [z[0] for z in list2D]
        listy = [z[1] for z in list2D]
        return (listx, listy)