import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from operator import add
import Correlation as corr_py
#import Interface as interf

from PIL import Image
import copy

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
#from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2TkAgg)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

from numba import jit, cuda 



"""
Class particle contains: 
	- the particle trajectory
	- particle color
	- information to which particle (particles) is it bound

"""
class Particle:
    def __init__ (self, color, start, stoch, diff1, diff2):
        self.color = color
        self.trajectory = start
        #self.trajectory.append(start)
        self.stoch = stoch
        self.diff = diff1
        self.diff_in_trap = diff2

class Trap:
    def __init__(self, R_t, start):
        self.R_t = R_t
        self.trajectory = start

"""
The simulation procedure:
	- all particles are placed randomly whithin simulation volume
	- 
"""

#(target ="cuda")
def Simulate (simulation_setup, All_particles, color_setup, trap_setup, ifpreview, frame00, win_width, win_height, dpi_all, root):
#def Simulate (Dim, N_g, N_r, N_y, D, T, width, height, depth, time_step, x0, y0, z0, x_res, y_res, z_res, figures_print, N_t, R_t, D_t, D_in_t, trapping):




    Dim = int(simulation_setup.Dimension)

    print ("Dimension: " + str(Dim))
    width = float(simulation_setup.width)
    height = float(simulation_setup.height)
    depth = float(simulation_setup.depth)

    x0 = width/2
    y0 = height/2
    z0 = depth/2

    image_counter = 0

    x__0 = x0
    y__0 = y0
    z__0 = z0

    x_res = float(simulation_setup.x_res)/1000
    y_res = float(simulation_setup.y_res)/1000
    z_res = float(simulation_setup.z_res)/1000

    figures_print = True

    T = float(simulation_setup.T)
    time_step = float(simulation_setup.time_step)
    dwell_time = float(simulation_setup.dwell_time)

    
    #scale_in = np.sqrt(2.0 * (float(D)) * float(time_step) / 1000)
    #scale_trap = np.sqrt(2.0 * (float(D_t)) * float(time_step) / 1000)
    #scale_in_trap = np.sqrt(2.0 * (float(D_in_t)) * float(time_step) / 1000)

    particles = []
    N = 0
    for particle in All_particles:
        N += int(particle.N)

    
    
        start_x = (np.random.uniform(0.0, 1.0, int(particle.N)))*width
        start_y = (np.random.uniform(0.0, 1.0, int(particle.N)))*height
        start_z = (np.random.uniform(0.0, 1.0, int(particle.N)))*depth

        if particle.St[0] == 0:
            color = color_setup.rp

        if particle.St[1] == 0:
            color = color_setup.gp

        if particle.St[1] > 0 and particle.St[0] > 0:
            color = "yellow"

    

        for i in range (0,int(particle.N)):
            if Dim == 3:
                particles.append(Particle(color, [float(start_x[i]), float(start_y[i]), float(start_z[i])], particle.St, particle.D1, particle.D2))

            if Dim == 2:
                particles.append(Particle(color, [float(start_x[i]), float(start_y[i])], particle.St, particle.D1, particle.D2))



    trapping = trap_setup.trapping
    D_t = float(trap_setup.D)
    R_t = float(trap_setup.R)
    N_t = int(trap_setup.N)
    print ("Trapping: ", trapping)
    #----------------------------------------------------------------------------------------------------------------------
    #-------------------------------Put traps------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    traps =[]
    if trapping == True:
        
        start_xx = np.linspace(0.2, 0.8, int(np.sqrt(N_t)))*width
        start_yy = np.linspace(0.2, 0.8, int(np.sqrt(N_t)) )*height
        #print (start_xx)
        #print (start_yy)
        start_x = []
        start_y = []
        for iii in start_xx:
            for jjj in start_yy:
                start_x.append(iii)
                start_y.append(jjj)



        traps=[]
    
        for i in range (0, N_t):
            traps.append(Trap(R_t, [float(start_x[i]), float(start_y[i])]))

    #----------------------------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------------------------
    
    colors = [particles[0].color]

    for p in particles:
        if (p.color in colors) == False:
            colors.append(p.color)

    steps = int(T*1000/time_step)
    #steps = 2


    numbers_g = []
    numbers_r = []

    if Dim == 3:
        mu = [x0, y0, z0]

    if Dim == 2:
        mu = [x0, y0]

    if Dim == 3:
        sigma = [x_res/2, y_res/2, z_res/2]

    if Dim == 2:
        sigma = [x_res/2, y_res/2]


    number_g = 0
    number_r = 0

        

    for p in particles:

        if Dim == 3:

            x = [p.trajectory[0], p.trajectory[1], p.trajectory[2]]

        if Dim == 2:

            x = [p.trajectory[0], p.trajectory[1]]

        number_g += p.stoch[0] * psf_3d (Dim, mu, sigma, x) 
        number_r += p.stoch[1] * psf_3d (Dim, mu, sigma, x) 

        
    

    numbers_g.append(number_g)
    numbers_r.append(number_r)


    """figure1 = Figure(figsize=(win_width/(2*dpi_all),win_height/(2*dpi_all)), dpi=100)
                figure1.patch.set_facecolor('black')
                gs = figure1.add_gridspec(2, 4)
            
                if Dim == 3:
                    #ax = p3.Axes3D(fig1)
                    box_plot = figure1.add_subplot(gs[0:2, 0:2], projection='3d')
            
                if Dim == 2:
                    box_plot = figure1.add_subplot(gs[0:2, 0:2])
            
                fluct_plot = figure1.add_subplot(gs[0, 2:4 ])
                corr_plot = figure1.add_subplot(gs[1, 2:4])
            
                
            
                canvas1 = FigureCanvasTkAgg(figure1, frame00)
                canvas1.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)
            
                toolbar = NavigationToolbar2Tk(canvas1, frame00)
                toolbar.update()
                canvas1.get_tk_widget().pack()
            
                figure1.tight_layout()"""
    figure1 = Figure(figsize=(win_width/(2*dpi_all),win_height/(2*dpi_all)), dpi=100)
    figure1.patch.set_facecolor('black')
    gs = figure1.add_gridspec(2, 4)

    if Dim == 3:
        #ax = p3.Axes3D(fig1)
        box_plot = figure1.add_subplot(gs[0:2, 0:2], projection='3d')

    if Dim == 2:
        box_plot = figure1.add_subplot(gs[0:2, 0:2])

    fluct_plot = figure1.add_subplot(gs[0, 2:4 ])
    corr_plot = figure1.add_subplot(gs[1, 2:4])

    

    canvas1 = FigureCanvasTkAgg(figure1, frame00)
    canvas1.get_tk_widget().pack(side = "top", anchor = "nw", fill="x", expand=True)

    toolbar = NavigationToolbar2Tk(canvas1, frame00)
    toolbar.update()
    canvas1.get_tk_widget().pack()

    figure1.tight_layout()

    for index in range (1,steps):

        

        #-------------------------------------------------------------------------------------------------
        #--------------------------------------Move traps-------------------------------------------------
        #-------------------------------------------------------------------------------------------------

        if trapping == True:

            scale_trap = np.sqrt(2.0 * (float(D_t)) * float(time_step) / 1000)
            dr = Direction_traps(Dim, N_t, scale_trap)


            for i in range (0,N_t):
                list1 = [sum(x11) for x11 in zip(traps[i].trajectory, dr[i])]

                

                traps[i].trajectory = boundary(Dim, list1, width, height, depth)
            
            
            traps = fusion_check(traps)

            N_t = len (traps)

        #print (particles[0].trajectory)
        
        #-------------------------------------------------------------------------------------------------
        #--------------------------------------Move particles---------------------------------------------
        #-------------------------------------------------------------------------------------------------
        
        dr = Direction(Dim, particles, width, height, traps, trapping, time_step)

    
        if (index+1) % 10000 == 0:
            print ("Completed timestep ", index+1, " of ", steps, "\r")
    
        for i in range (0,N):


            
            list1 = [sum(x11) for x11 in zip(particles[i].trajectory, dr[i])]
            
            particles[i].trajectory = boundary(Dim, list1, width, height, depth)

        #print (particles[0].trajectory)
        

        number_g = 0
        number_r = 0

        

        for p in particles:

            if Dim == 3:

                x = [p.trajectory[0], p.trajectory[1], p.trajectory[2]]

            if Dim == 2:

                x = [p.trajectory[0], p.trajectory[1]]

            number_g += p.stoch[0] * psf_3d (Dim, mu, sigma, x)
            number_r += p.stoch[1] * psf_3d (Dim, mu, sigma, x)

        
    

        numbers_g.append(number_g)
        numbers_r.append(number_r)


        
        if (index-1)*1000%steps == 0 and figures_print == True:

            box_plot.cla()
            fluct_plot.cla()
            corr_plot.cla()

            print ("Printed timestep ", index+1, " of ", steps, "\r")

            #---------------------------------------------------------------------------------------------------------
            #----------------------------Print particles--------------------------------------------------------------
            #---------------------------------------------------------------------------------------------------------

            box_plot.set_title("Simulation box", color="white")



            #--------------------------------------Print traps-------------------------------------------------

            if trapping == True:



                for tr in traps:

                    x0 = tr.trajectory[0]
                    y0 = tr.trajectory[1]

                    draw_circle = plt.Circle((x0, y0), tr.R_t, color = "black", linewidth = 0, alpha = 0.5)
                    box_plot.add_artist(draw_circle)


                for tr in traps:
                    x0 = tr.trajectory[0]
                    y0 = tr.trajectory[1] + height

                    draw_circle = plt.Circle((x0, y0), tr.R_t, color = "black", linewidth = 0, alpha = 0.5)
                    box_plot.add_artist(draw_circle)


                for tr in traps:
                    x0 = tr.trajectory[0]
                    y0 = tr.trajectory[1] - height

                    draw_circle = plt.Circle((x0, y0), tr.R_t, color = "black", linewidth = 0, alpha = 0.5)
                    box_plot.add_artist(draw_circle)

    


                for tr in traps:    
                    x0 = tr.trajectory[0] + width
                    y0 = tr.trajectory[1]

                    draw_circle = plt.Circle((x0, y0), tr.R_t, color = "black", linewidth = 0, alpha = 0.5)
                    box_plot.add_artist(draw_circle)


    
                for tr in traps:
                    x0 = tr.trajectory[0] + width
                    y0 = tr.trajectory[1] + height

                    draw_circle = plt.Circle((x0, y0), tr.R_t, color = "black", linewidth = 0, alpha = 0.5)
                    box_plot.add_artist(draw_circle)



                for tr in traps:
                    x0 = tr.trajectory[0] + width
                    y0 = tr.trajectory[1] - height

                    draw_circle = plt.Circle((x0, y0), tr.R_t, color = "black", linewidth = 0, alpha = 0.5)
                    box_plot.add_artist(draw_circle)




    
                for tr in traps:
                    x0 = tr.trajectory[0] - width
                    y0 = tr.trajectory[1]

                    draw_circle = plt.Circle((x0, y0), tr.R_t, color = "black", linewidth = 0, alpha = 0.5)
                    box_plot.add_artist(draw_circle)



                for tr in traps:
                    x0 = tr.trajectory[0] - width
                    y0 = tr.trajectory[1] + height

                    draw_circle = plt.Circle((x0, y0), tr.R_t, color = "black", linewidth = 0, alpha = 0.5)
                    box_plot.add_artist(draw_circle)



                for tr in traps:
                    x0 = tr.trajectory[0] - width
                    y0 = tr.trajectory[1] - height

    
                    draw_circle = plt.Circle((x0, y0), tr.R_t, color = "black", linewidth = 0, alpha = 0.5)
                    box_plot.add_artist(draw_circle)



            #--------------------------------------Print particles---------------------------------------------

            if Dim == 3:
        

                for color in colors:

                    particles_x = []
                    particles_y = []
                    particles_z = []

                    for p in particles:
                        if (p.color == color):

                            particles_x.append(p.trajectory[0])
                            particles_y.append(p.trajectory[1])
                            particles_z.append(p.trajectory[2])


                    box_plot.set_xlim(0, width)
                    box_plot.set_ylim(0, height) 
                    box_plot.set_zlim(0, depth) 
                    if (color != 'yellow'):
                        box_plot.scatter(particles_x, particles_y, particles_z, s = 50, linewidth = 0, color = color)

                    if (color == 'yellow'):
                        box_plot.scatter(particles_x, particles_y, particles_z, s = 50, linewidth = 1, facecolor='g', edgecolor = 'magenta')
             
    

            


            if Dim == 2:
        

                for color in colors:

                    particles_x = []
                    particles_y = []

                    for p in particles:
                        if (p.color == color):

                            particles_x.append(p.trajectory[0])
                            particles_y.append(p.trajectory[1])
                            


                    box_plot.set_xlim(0, width)
                    box_plot.set_ylim(0, height) 
                     
                    if (color != 'yellow'):
                        box_plot.scatter(particles_x, particles_y, s = 50, linewidth = 0, color = color)

                    if (color == 'yellow'):
                        box_plot.scatter(particles_x, particles_y, s = 50, linewidth = 1, facecolor='g', edgecolor = 'magenta')

            box_plot.spines['bottom'].set_color('white')
            box_plot.spines['top'].set_color('white') 
            box_plot.spines['right'].set_color('white')
            box_plot.spines['left'].set_color('white')
            box_plot.tick_params(axis='x', colors='white')
            box_plot.tick_params(axis='y', colors='white')
            if Dim == 3:
                box_plot.tick_params(axis='z', colors='white')
                box_plot.zaxis.label.set_color('white')
            box_plot.yaxis.label.set_color('white')
            box_plot.xaxis.label.set_color('white')
            box_plot.set_facecolor('black')
             
            
            #--------------------------------------Print focal point---------------------------------------------
            

            Plot_focal_point (Dim, box_plot, x__0, y__0, z__0, x_res, y_res, z_res)
            if (Dim == 2):
                box_plot.set_aspect(1)

            #---------------------------------------------------------------------------------------------------------
            #----------------------------Print traces-----------------------------------------------------------------
            #---------------------------------------------------------------------------------------------------------

            fluct_plot.set_title("Intensity traces", color="white")

            x_length = 0.5

            t1 = np.linspace (0,index, index+1)
            t1 = t1*time_step/1000


            if t1[len(t1)-1] <= x_length:
                fluct_plot.set_xlim(0, x_length)

            if t1[len(t1)-1] > x_length:
                #fluct_plot.set_xlim(t1[len(t1)-1] - x_length, t1[len(t1)-1])
                fluct_plot.set_xlim(0, t1[len(t1)-1])



            
            fluct_plot.plot(t1,numbers_g,'g')
            fluct_plot.plot(t1,numbers_r, color = 'magenta')
            fluct_plot.ticklabel_format(axis = "y", style="sci", scilimits = (0,0))
            fluct_plot.set_ylabel('Intensity (a.u.)')
            fluct_plot.set_xlabel('Time (s)')

            fluct_plot.spines['bottom'].set_color('white')
            fluct_plot.spines['top'].set_color('white') 
            fluct_plot.spines['right'].set_color('white')
            fluct_plot.spines['left'].set_color('white')
            fluct_plot.tick_params(axis='x', colors='white')
            fluct_plot.tick_params(axis='y', colors='white')
            fluct_plot.yaxis.label.set_color('white')
            fluct_plot.xaxis.label.set_color('white')
            fluct_plot.set_facecolor('black')

            #---------------------------------------------------------------------------------------------------------
            #----------------------------Print correlation functions--------------------------------------------------
            #---------------------------------------------------------------------------------------------------------

            corr_plot.set_title("Correlation functions", color="white")

            if index > 50:

                time_step_c = (t1[1] - t1[0])/1000

    


                # Green autocorrelation
                a = np.array(numbers_g)
                b = np.array(numbers_g)

    
                corr = corr_py.correlate_linear(a,b)#correlate linear
                #Calculates smoothed version of curve using logarithmic bins.
                bins = corr_py.logbins(a.size//2, 64)
                scorr = corr_py.smooth(corr_py.binaver(corr,bins))
                #scorr = corr_py.binaver(corr,bins)
                time = bins*time_step_c

                #Plotting
    

                #plt.semilogx(np.arange(0,a.shape[0]//2)*time_step,corr,'g')
                corr_plot.semilogx(time,scorr,'g')

                # Red autocorrelation
                a = np.array(numbers_r)
                b = np.array(numbers_r)

    
                corr = corr_py.correlate_linear(a,b)#correlate linear
                #Calculates smoothed version of curve using logarithmic bins.
                bins = corr_py.logbins(a.size//2, 64)
                scorr = corr_py.smooth(corr_py.binaver(corr,bins))
                #scorr = corr_py.binaver(corr,bins)
                time = bins*time_step_c

                #Plotting
    

                #plt.semilogx(np.arange(0,a.shape[0]//2)*time_step,corr,'r')
                corr_plot.semilogx(time,scorr,color = 'magenta')


                # Cross correlation
                a = np.array(numbers_g)
                b = np.array(numbers_r)

    
                corr = corr_py.correlate_linear(a,b)#correlate linear
                #Calculates smoothed version of curve using logarithmic bins.
                bins = corr_py.logbins(a.size//2, 64)
                scorr = corr_py.smooth(corr_py.binaver(corr,bins))
                #scorr = corr_py.binaver(corr,bins)
                time = bins*time_step_c

                #Plotting
    

                #plt.semilogx(np.arange(0,a.shape[0]//2)*time_step,corr,'y')
                corr_plot.semilogx(time,scorr,'y')

                #plt.xlim (10**(-9), 10**(-6))

            corr_plot.autoscale()
            corr_plot.set_ylabel('G(tau)')
            corr_plot.set_xlabel('Tau (s)')

            corr_plot.spines['bottom'].set_color('white')
            corr_plot.spines['top'].set_color('white') 
            corr_plot.spines['right'].set_color('white')
            corr_plot.spines['left'].set_color('white')
            corr_plot.tick_params(axis='x', colors='white')
            corr_plot.tick_params(axis='y', colors='white')
            corr_plot.yaxis.label.set_color('white')
            corr_plot.xaxis.label.set_color('white')
            corr_plot.set_facecolor('black')



            
            if image_counter == 1:
                box_plot.set_position(pos1)
                fluct_plot.set_position(pos2)
                corr_plot.set_position(pos3)
            
            if image_counter == 0:
                figure1.tight_layout()

            
                pos1 = box_plot.get_position()
                pos2 = fluct_plot.get_position()
                pos3 = corr_plot.get_position()

                image_counter = 1
    
            
            """
            if image_counter == 1:

                fname = 'E:\\Scripts\\FCCS as FACS\\Simulation of diffusion\\2020.08.27 - 2D and Trap\\Figures\\Movie1.tif'
                plt.savefig(fname, dpi=200, format="tif",
                        transparent=False, bbox_inches=None, pad_inches=0.1,
                        frameon=None, metadata=None)


                im1 = Image.open('E:\\Scripts\\FCCS as FACS\\Simulation of diffusion\\2020.08.27 - 2D and Trap\\Figures\\Movie.tif')
                im2 = Image.open('E:\\Scripts\\FCCS as FACS\\Simulation of diffusion\\2020.08.27 - 2D and Trap\\Figures\\Movie1.tif')

                im3 = copy.deepcopy(im1)
                im4 = copy.deepcopy(im2)

                im1.close()
                im2.close()

                out = 'E:\\Scripts\\FCCS as FACS\\Simulation of diffusion\\2020.08.27 - 2D and Trap\\Figures\\Movie.tif'

                im3.save(out, save_all=True, append_images=[ im4])


            if image_counter == 0:

                fname = 'E:\\Scripts\\FCCS as FACS\\Simulation of diffusion\\2020.08.27 - 2D and Trap\\Figures\\Movie.tif'
                plt.savefig(fname, dpi=200, format="tif",
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)

                image_counter = 1

            
            """
            
            
            #fname = 'C:\\Users\\taras.sych\\Science\\Program development\\Simulation of diffusion\\2020.09.09 - Interface\\Output\\figure' + str(index) + '.tif'
            #fname = 'C:\\Users\\taras.sych\\Desktop\\Sim output\\Full cross corr\\figure' + str(index) + '.tif'
            fname = 'C:\\Users\\taras.sych\\Desktop\\Sim output\\figure' + str(index) + '.tif'
            #fname = 'C:\\Users\\taras.sych\\Desktop\\Sim output\\No cross corr\\figure' + str(index) + '.tif'
            figure1.savefig(fname, dpi=200, format="tif",transparent=False, facecolor = 'black', bbox_inches=None, pad_inches=0.1, metadata=None)
            figure1.patch.set_facecolor('black')
            
            canvas1.draw()
            figure1.tight_layout()
            
            
            #plt.close(figure1)
           
        

        root.update()   
        
    """        
    f= open("Brownian_3D.txt","w+")

    f.write("time\tx\ty\tz\n")

    

    print ("Saving trajectories in Brownian_3D.txt ...")

    for index in range(0,N):
    
        f.write("Particle: " + str(index) + "\n" )
        f.write("Color: " + particles[index].color + "\n" )
    
        for j in range(len(particles[i].trajectory)):

            f.write(str(j) + "\t" + str(particles[i].trajectory[j][0]) + "\t" + str(particles[i].trajectory[j][1]) + "\t" + str(particles[i].trajectory[j][2]) + "\n")

    f.close()
    """
    
    

    
    return [numbers_g, numbers_r]





"""
Boundary condition:
- in case particle drifts outside of the simulation volume, it appears on the other side of the simulation
"""
def boundary (Dim, r, width, height, depth):

    
    r1 = []
    
    if Dim == 3:
        r1.append(r[0]%width)
        r1.append(r[1]%height)
        r1.append(r[2]%depth)

    if Dim == 2:
        r1.append(r[0]%width)
        r1.append(r[1]%height)
    
    return r1

"""
Generation of the random direction for random walk algorithm
"""
def Direction_traps (Dim, N, scale_in):
    phi = np.random.uniform (0, 1, N)*np.pi
    theta = np.random.uniform (-1, 1, N)*np.pi
    
    if Dim == 3:
        dx = (scale_in*np.sin(phi)*np.cos(theta)).tolist()
        dy = (scale_in*np.sin(phi)*np.sin(theta)).tolist()
        dz = scale_in*np.cos(phi)
        dz = dz.tolist()
    
    
    
        arr = np.array([dx,dy,dz])


        arr1 = arr.T.tolist()

    if Dim == 2:
        dx = (scale_in*np.cos(theta)).tolist()
        dy = (scale_in*np.sin(theta)).tolist() 
        arr = np.array([dx,dy])

        arr1 = arr.T.tolist() 
    
    return arr1

def Direction(Dim, ps, width, height, traps, trapping, time_step):

    


    phi = np.random.uniform (0, 1, len(ps))*np.pi
    theta = np.random.uniform (-1, 1, len(ps))*np.pi
    dx = [None] * len(phi)
    dy = [None] * len(phi)
    dz = [None] * len(phi)



    trapped1 = False
    
    if Dim == 3:
        for i in range(len(ps)):
            if trapping == True:
                trapped1 = trapped(traps, width, height, ps[i].trajectory)

            if trapped1 == True:
                D = float(ps[i].diff_in_trap)
                scale_in = np.sqrt(2.0 * (float(D)) * float(time_step) / 1000)

            if trapped1 == False:
                D = float(ps[i].diff)
                scale_in = np.sqrt(2.0 * (float(D)) * float(time_step) / 1000)
            


            dx[i] = scale_in*np.sin(phi[i])*np.cos(theta[i])
            dy[i] = scale_in*np.sin(phi[i])*np.sin(theta[i])
            dz[i] = scale_in*np.cos(phi[i])
    
    
        
        arr = np.array([dx,dy,dz])


        arr1 = arr.T.tolist()

    if Dim == 2:
        for i in range(len(ps)):
            
            if trapping == True:
                trapped1 = trapped(traps, width, height, ps[i].trajectory)
            if trapped1 == True:

                D = float(ps[i].diff_in_trap)
                scale_in = np.sqrt(2.0 * (float(D)) * float(time_step) / 1000)

            if trapped1 == False:
                
                D = float(ps[i].diff)
                scale_in = np.sqrt(2.0 * (float(D)) * float(time_step) / 1000)
            dx[i] = scale_in*np.cos(theta[i])
            dy[i] = scale_in*np.sin(theta[i])
        
        arr = np.array([dx,dy])

        arr1 = arr.T.tolist() 
    
    return arr1




"""
Primitive "inside ellipsoide" intensity integration

"""

def Integrate_intensity (particles, x0, y0, z0, x_res, y_res, z_res, steps):
    
    print("Integrating intensity")

    numbers_g = []
    numbers_r = []
    
    mu = [x0, y0, z0]

    sigma = [x_res/2, y_res/2, z_res/2]

    
    
    for index in range (0, steps):
        
        if (index+1) % 10000 == 0:
            print ("Completed timestep ", index+1, " of ", steps, "\r")
        
        number_g = 0
        number_r = 0

        for p in particles:

            x = [p.trajectory[index][0], p.trajectory[index][1], p.trajectory[index][2]]

            number_g += p.stoch[0] * psf_3d (mu, sigma, x) 
            number_r += p.stoch[1] * psf_3d (mu, sigma, x)


        
        """
        for p in particles:
            x = p.trajectory[index][0] 
            y = p.trajectory[index][1]
            z = p.trajectory[index][2]

            psf_3d(mu, sigma, x)
            
            #if ((x-x0)**2/(x_res**2) + (y-y0)**2/(y_res**2) + (z-z0)**2/(z_res**2)) < 1 :
                #number_g += p.stoch[0]
                #number_r += p.stoch[1]
        """



        numbers_g.append(number_g)
        numbers_r.append(number_r)

        # Sample over dwell times


        
    return [numbers_g, numbers_r]


def psf_3d (Dim, mu, sigma, x):
    
    E = np.zeros ((len(sigma),len(sigma)))

    for i in range (len(sigma)):
        E [i][i] = sigma[i]**2


    E_inv = np.linalg.inv(E)

    x = np.array(x)
    mu = np.array(mu)

    A = 1/np.sqrt( ((2*np.pi)**Dim) * np.linalg.det(E))


    return A*np.exp(-0.5*np.dot(np.dot((x-mu),E_inv),(x-mu).T)) 










def Read_brownian (filepath):

    f = open(filepath)

    lines = f.readlines()

    indices = []


    for i in range (len(lines)):

        if "Particle" in lines[i]:
            indices.append(i)

    indices.append(len(lines))


    particles = []

    for i in range (len(indices)-1):
        str1, color = lines[indices[i]+1].split (" ")
        particle = Particle(color, [], "non-bound")

        print ("Particle ", i)

        for j in range (indices[i] + 2, indices[i+1]-1):
            
            particle.trajectory.append([float(lines[j].split ("\t")[1]), float(lines[j].split ("\t")[2]), float(lines[j].split ("\t")[3])])

        particles.append(particle)


    return particles


def Plot_focal_point (Dim, ax, x0, y0, z0, rx, ry, rz):

    if Dim == 3:

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
        x = rx/2 * np.outer(np.cos(u), np.sin(v))+x0
        y = ry/2 * np.outer(np.sin(u), np.sin(v))+y0
        z = rz/2 * np.outer(np.ones_like(u), np.cos(v))+z0

# Plot:
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)

    if Dim == 2:
        t = np.linspace(0,360,360)
        x = rx/2*np.cos(np.radians(t))+x0 # major axis of ellipse
        y = ry/2*np.sin(np.radians(t))+y0
        ax.plot(x,y, color='b')
        
    

# Adjustment of the axes, so that they all have the same span:
def Save_traces(t, intensities, conc):

    f= open("Intensity traces.txt","w+")

    f.write("time\tgreen channel\tred channel\n")

    int1 = intensities[0]
    int2 = intensities[1]

    for i in range(len(t)):
    
        f.write(str(t[i]) + "\t" + str(int1[i]) + "\t" + str(int2[i]) + "\n" )
        


def trapped(traps, width, height, x):
    flag = False



    for tr in traps:

        x0 = tr.trajectory[0]
        y0 = tr.trajectory[1]
        R_t = tr.R_t

    
        if (x[0] - x0)**2 + (x[1] - y0)**2 < R_t**2:
            flag = True


    for tr in traps:
        x0 = tr.trajectory[0]
        y0 = tr.trajectory[1] + height
        R_t = tr.R_t

    
        if (x[0] - x0)**2 + (x[1] - y0)**2 < R_t**2:
            flag = True

    for tr in traps:
        x0 = tr.trajectory[0]
        y0 = tr.trajectory[1] - height
        R_t = tr.R_t

    
        if (x[0] - x0)**2 + (x[1] - y0)**2 < R_t**2:
            flag = True

    for tr in traps:    
        x0 = tr.trajectory[0] + width
        y0 = tr.trajectory[1]
        R_t = tr.R_t

    
        if (x[0] - x0)**2 + (x[1] - y0)**2 < R_t**2:
            flag = True

    
    for tr in traps:
        x0 = tr.trajectory[0] + width
        y0 = tr.trajectory[1] + height
        R_t = tr.R_t

    
        if (x[0] - x0)**2 + (x[1] - y0)**2 < R_t**2:
            flag = True

    for tr in traps:
        x0 = tr.trajectory[0] + width
        y0 = tr.trajectory[1] - height
        R_t = tr.R_t

    
        if (x[0] - x0)**2 + (x[1] - y0)**2 < R_t**2:
            flag = True


    
    for tr in traps:
        x0 = tr.trajectory[0] - width
        y0 = tr.trajectory[1]
        R_t = tr.R_t

    
        if (x[0] - x0)**2 + (x[1] - y0)**2 < R_t**2:
            flag = True

    for tr in traps:
        x0 = tr.trajectory[0] - width
        y0 = tr.trajectory[1] + height
        R_t = tr.R_t

    
        if (x[0] - x0)**2 + (x[1] - y0)**2 < R_t**2:
            flag = True

    for tr in traps:
        x0 = tr.trajectory[0] - width
        y0 = tr.trajectory[1] - height
        R_t = tr.R_t

    
        if (x[0] - x0)**2 + (x[1] - y0)**2 < R_t**2:
            flag = True
    

   

    return flag

def fusion_check(traps):
        

    i=0;
    

    while i<len(traps)-1:
        
        
        j=i+1
        flag = False

        while j < len(traps) and flag == False:

            
                
            x0 = traps[i].trajectory[0]
            y0 = traps[i].trajectory[1]
            x1 = traps[j].trajectory[0]
            y1 = traps[j].trajectory[1]
            R0 = traps[i].R_t
            R1 = traps[j].R_t



            if (x1 - x0)**2 + (y1 - y0)**2 < (0.8*(R0+R1))**2:
                flag == True
                    
                x3 = (x1+x0)/2
                y3 = (y1+y0)/2
                R3 = np.sqrt(R0**2 + R1**2)
                del traps[max(i,j)]
                del traps[min(i,j)]
                traps.append(Trap(R3, [x3, y3]))
                i=0

           

            j+=1

        i+=1

    return traps



