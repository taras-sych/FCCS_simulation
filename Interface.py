import tkinter as tk
from tkinter import ttk
from tkinter import font as tkFont
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
#from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2TkAgg)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

from ttkwidgets import CheckboxTreeview
import codecs
import os
from scipy import stats
import copy
import numpy as np

import Simulation as sim_py

import Correlation as corr_py

from tkinter import colorchooser

dpi_all = 100

All_particles = []

selected_item = "not selected"
selected_item_index = "not selected"

global iftrap
iftrap = False
global iftrapmove
iftrapmove = False

global Dim
Dim = 3



def Run():
	global Dim
	#try:
	ifpreview = False
	print("Dimension:" + str(Dim))
	simulation_setup = Sim_box(Dim, X_size.get(), Y_size.get(), Z_size.get(), X_resol.get(), Y_resol.get(), Z_resol.get(), Sim_time.get(), Step_time.get(), Dwell_time.get())
	color_setup = Plot_colors(PGreencolor_entry.get(), PRedcolor_entry.get(), TGreencolor_entry.get(), TRedcolor_entry.get(), CGreencolor_entry.get(), CRedcolor_entry.get(), Cross_entry.get(), Focal_entry.get())
	trap_setup = Trap_values(Traps_number_entry.get(), Traps_radii_entry.get(), Traps_diff_entry.get(), iftrap, iftrapmove)

	traces = sim_py.Simulate (simulation_setup, All_particles, color_setup, trap_setup, ifpreview, frame00, win_width, win_height, dpi_all, root)
	#except:
		#tk.messagebox.showerror(title="Sorry!", message="I don't feel like doing simulation right now")

def Select_row(event):
	selected_item = particle_list.selection()[0]
	num1, selected_item_index = particle_list.selection()[0].split('I')
	selected_item_index = int(selected_item_index, 16)
	print (selected_item_index)
	print (selected_item)

class Particles_overview:
	def __init__(self, N1, St1, D11, D21):
		self.N = N1
		self.St = St1
		self.D1 = D11
		self.D2 = D21 

class Plot_colors:
	def __init__(self, gp1, rp1, gt1, rt1, gc1, rc1, cc1, fc1 ):
		self.gp = gp1
		self.rp = rp1
		self.gt = gt1
		self.rt = rt1
		self.gc = gc1
		self.rc = rc1
		self.cc = cc1
		self.fc = fc1

	def color_print(self):
		print(self.gp, self.rp, self.gt, self.rt, self.gc, self.rc, self.cc, self.fc)

class Sim_box:
	def __init__(self, Dim1, x1, y1, z1, xres, yres, zres, T1, time_step1, dwell_time1):
		self.Dimension = Dim1
		self.width = x1
		self.height = y1
		self.depth = z1
		self.x_res = xres
		self.y_res = yres
		self.z_res = zres
		self.T = T1
		self.time_step = time_step1
		self.dwell_time = dwell_time1

class Trap_values:
	def __init__(self, N1, radius1, D1111, trapping1, moving1):
		self.trapping = trapping1
		self.moving = moving1
		self.N = N1
		self.R = radius1
		self.D = D1111





def Check_trap():
	global iftrap



	if var1.get() == 0:
		iftrap = False
		Traps_number_entry.config(state = "disabled")
		Traps_diff_entry.config(state = "disabled")
		Traps_radii_entry.config(state = "disabled")
		Check_trap_diff_button.config(state = "disabled")
		var2.set(0)

	if var1.get() == 1:
		iftrap = True
		Traps_number_entry.config(state = "normal")
		Traps_diff_entry.config(state = "normal")
		Traps_radii_entry.config(state = "normal")
		Check_trap_diff_button.config(state = "normal")
		var2.set(1)




	

def Check_trap_diff():
	global iftrapmove

	if var2.get() == 0:
		iftrapmove = False
		Traps_diff_entry.config(state = "disabled")
		

	if var2.get() == 1:
		iftrapmove = True
		Traps_diff_entry.config(state = "normal")


def Import():
	plot_col.color_print()


def Set_dimensions(event):
	global Dim
	temp = Dimensions.get()
	if temp == "2D":
		Dim = 2
		Z_resol.config(state='disabled')
		Z_size.config(state='disabled')
		Check_trap_button.config(state = "normal")
		var1.set(0)
		Traps_number_entry.config(state = "disabled")
		Traps_diff_entry.config(state = "disabled")
		Traps_radii_entry.config(state = "disabled")
		Check_trap_diff_button.config(state = "disabled")
		var2.set(0)

	if temp == "3D":
		Dim = 3
		Z_resol.config(state='normal')
		Z_size.config(state='normal')
		Check_trap_button.config(state = "disabled")
		Traps_number_entry.config(state = "disabled")
		Traps_diff_entry.config(state = "disabled")
		Traps_radii_entry.config(state = "disabled")
		Check_trap_diff_button.config(state = "disabled")
		var1.set(0)
		var2.set(0)




class Add_frame:
	def __init__(self):
		self.add_particle = tk.Toplevel()
		self.add_particle.title("Add particles")

		self.N_label = tk.Label(self.add_particle, text="Number of particles: ")
		self.N_label.grid(row = 0, column = 0, sticky = "w")

		self.N_entry = tk.Entry(self.add_particle, width = 9)
		self.N_entry.grid(row = 0, column = 1, sticky = "w")
		self.N_entry.delete(0,"end")
		self.N_entry.insert(0,"3")

		self.Green_label = tk.Label(self.add_particle, text="Green units: ")
		self.Green_label.grid(row = 1, column = 0, sticky = "w")

		self.Green_entry = tk.Entry(self.add_particle, width = 9)
		self.Green_entry.grid(row = 1, column = 1, sticky = "w")
		self.Green_entry.delete(0,"end")
		self.Green_entry.insert(0,"1")

		self.Red_label = tk.Label(self.add_particle, text="Red units: ")
		self.Red_label.grid(row = 2, column = 0, sticky = "w")

		self.Red_entry = tk.Entry(self.add_particle, width = 9)
		self.Red_entry.grid(row = 2, column = 1, sticky = "w")
		self.Red_entry.delete(0,"end")
		self.Red_entry.insert(0,"1")

		self.Diff_label = tk.Label(self.add_particle, text="Diffusion coefficient (um^2/s): ")
		self.Diff_label.grid(row = 3, column = 0, sticky = "w")

		self.Diff_entry = tk.Entry(self.add_particle, width = 9)
		self.Diff_entry.grid(row = 3, column = 1, sticky = "w")
		self.Diff_entry.delete(0,"end")
		self.Diff_entry.insert(0,"0.1")

		self.Diff_trap_label = tk.Label(self.add_particle, text="Diffusion coefficient when in trap (um^2/s): ")
		self.Diff_trap_label.grid(row = 4, column = 0, sticky = "w")

		self.Diff_trap_entry = tk.Entry(self.add_particle, width = 9)
		self.Diff_trap_entry.grid(row = 4, column = 1, sticky = "w")
		self.Diff_trap_entry.delete(0,"end")
		self.Diff_trap_entry.insert(0,"0.01")

		self.OK_Button = tk.Button(self.add_particle, text="OK", command=self.Add_to_table)
		self.OK_Button.grid(row = 5, column = 0, columnspan = 2)

		self.add_particle.grab_set()



	def Add_to_table(self):
		try:
			new_element = Particles_overview(float(self.N_entry.get()), [int(self.Green_entry.get()), int(self.Red_entry.get())], float(self.Diff_entry.get()), float(self.Diff_trap_entry.get()))
			All_particles.append(new_element)
			particle_list.insert('', 'end', values=(self.N_entry.get(), self.Green_entry.get(), self.Red_entry.get(), self.Diff_entry.get(), self.Diff_trap_entry.get() ))
			self.add_particle.destroy()
			selected_item = "not selected"
			selected_item_index = "not selected"
		except:
			tk.messagebox.showerror(title="Error", message="Numbers only")
			print("only numbers!")
		




def Add():
	new_window = Add_frame()


def Edit():
	add_particle = tk.Toplevel()

	N_label = tk.Label(add_particle, text="Number: ")
	N_label.grid(row = 0, column = 0, sticky = "w")

	N_entry = tk.Entry(add_particle, width = 9)
	N_entry.grid(row = 0, column = 1, sticky = "w")

	Green_label = tk.Label(add_particle, text="Green: ")
	Green_label.grid(row = 1, column = 0, sticky = "w")

	Green_entry = tk.Entry(add_particle, width = 9)
	Green_entry.grid(row = 1, column = 1, sticky = "w")

	Red_label = tk.Label(add_particle, text="Red: ")
	Red_label.grid(row = 2, column = 0, sticky = "w")

	Red_entry = tk.Entry(add_particle, width = 9)
	Red_entry.grid(row = 2, column = 1, sticky = "w")

	Diff_label = tk.Label(add_particle, text="Diffusion: ")
	Diff_label.grid(row = 3, column = 0, sticky = "w")

	Diff_entry = tk.Entry(add_particle, width = 9)
	Diff_entry.grid(row = 3, column = 1, sticky = "w")

	Diff_trap_label = tk.Label(add_particle, text="Diff trap: ")
	Diff_trap_label.grid(row = 4, column = 0, sticky = "w")

	Diff_trap_entry = tk.Entry(add_particle, width = 9)
	Diff_trap_entry.grid(row = 4, column = 1, sticky = "w")

	OK_Button = tk.Button(add_particle, text="OK", command=Import)
	OK_Button.grid(row = 5, column = 0, columnspan = 2)


def Remove():
	#try:
	particle_list.delete(selected_item)
	#except:
		#print("Select item to delete")


def PR_color():
	plot_col.gp = colorchooser.askcolor()[1]

	PRedcolor_entry.config(state='normal')
	PRedcolor_entry.delete(0,"end")
	PRedcolor_entry.insert(0,plot_col.gp)
	PRedcolor_entry.config(readonlybackground  = plot_col.gp)
	PRedcolor_entry.config(state='readonly')


def PG_color():
	plot_col.rp = colorchooser.askcolor()[1]
	PGreencolor_entry.config(state='normal')
	PGreencolor_entry.delete(0,"end")
	PGreencolor_entry.insert(0,str(plot_col.rp))
	PGreencolor_entry.config(readonlybackground  = plot_col.rp)
	PGreencolor_entry.config(state='readonly')

def TR_color():
	plot_col.rt = colorchooser.askcolor()[1]
	TRedcolor_entry.config(state='normal')
	TRedcolor_entry.delete(0,"end")
	TRedcolor_entry.insert(0,str(plot_col.rt))
	TRedcolor_entry.config(readonlybackground  = plot_col.rt)
	TRedcolor_entry.config(state='readonly')

def TG_color():
	plot_col.gt = colorchooser.askcolor()[1]
	TGreencolor_entry.config(state='normal')
	TGreencolor_entry.delete(0,"end")
	TGreencolor_entry.insert(0,str(plot_col.gt))
	TGreencolor_entry.config(readonlybackground  = plot_col.gt)
	TGreencolor_entry.config(state='readonly')

def CR_color():
	plot_col.rc = colorchooser.askcolor()[1]
	CRedcolor_entry.config(state='normal')
	CRedcolor_entry.delete(0,"end")
	CRedcolor_entry.insert(0,str(plot_col.rc))
	CRedcolor_entry.config(readonlybackground  = plot_col.rc)
	CRedcolor_entry.config(state='readonly')

def CG_color():
	plot_col.gc = colorchooser.askcolor()[1]
	CGreencolor_entry.config(state='normal')
	CGreencolor_entry.delete(0,"end")
	CGreencolor_entry.insert(0,str(plot_col.gc))
	CGreencolor_entry.config(readonlybackground  = plot_col.gc)
	CGreencolor_entry.config(state='readonly')

def CC_color():
	plot_col.cc = colorchooser.askcolor()[1]
	Cross_entry.config(state='normal')
	Cross_entry.delete(0,"end")
	Cross_entry.insert(0,str(plot_col.cc))
	Cross_entry.config(readonlybackground  = plot_col.cc)
	Cross_entry.config(state='readonly')

def FS_color():
	plot_col.fs = colorchooser.askcolor()[1]
	Focal_entry.config(state='normal')
	Focal_entry.delete(0,"end")
	Focal_entry.insert(0,str(plot_col.fs))
	Focal_entry.config(readonlybackground  = plot_col.fs)
	Focal_entry.config(state='readonly')




root = tk.Tk()
root.title("Diffusion simulation")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

win_width = round(0.8 * screen_width)
win_height = round (0.8 * screen_height)


line = str(win_width) + "x" + str(win_height)


root.geometry(line)

dpi_all = 100


frame0_width = round(win_width/4)
frame0 = tk.LabelFrame(root)
frame0.pack(side = "left", anchor = "nw", expand =1, fill=tk.BOTH)
frame0.config(bd=0, width = frame0_width, height = win_height)
frame0.grid_propagate(0)


frame01 = tk.Frame(frame0)
frame01.pack(side="top", fill="x")

Import_config_Button = tk.Button(frame01, text="Import config", command=Import)
Import_config_Button.pack(side = "left", anchor = "nw")



Save_config_Button = tk.Button(frame01, text="Save config", command=Import)
Save_config_Button.pack(side = "left", anchor = "nw")

Config_name = tk.Entry(frame01, width = 30)
Config_name.pack(side = "left", anchor = "nw")

tabs = ttk.Notebook(frame0, width=round(win_width/4), height = round(win_height/2), padding = 0)

#tab = []


frame1 = ttk.Frame(tabs)
frame2 = ttk.Frame(tabs)
frame3 = ttk.Frame(tabs)
frame4 = ttk.Frame(tabs)
frame5 = ttk.Frame(tabs)

tabs.add(frame1, text = "Simulation box")
#------------------------------------------------------------------------------------------------
#----------------------------         Simulation box      ---------------------------------------
#------------------------------------------------------------------------------------------------
row11 = ttk.Frame(frame1)
row11.pack(side="top", fill="x")

Dim_label = tk.Label(row11, text="Dimensions: ")
Dim_label.grid(row = 0, column = 0, sticky = "w")


Dimensions = ttk.Combobox(row11,values = ["3D", "2D"], width = 10 )
Dimensions.config(state = "readonly")
Dimensions.grid(row = 0, column = 1, sticky = "w")
Dimensions.set("3D")
Dimensions.bind("<<ComboboxSelected>>", Set_dimensions)

row21 = ttk.Frame(frame1)
row21.pack(side="top", fill="x")
Size_label = tk.Label(row21, text="Size (micron): ")
Size_label.grid(row = 0, column = 0, sticky = "w")

row31 = ttk.Frame(frame1)
row31.pack(side="top", fill="x")
Xs_label = tk.Label(row31, text="X: ")
Xs_label.grid(row = 0, column = 0, sticky = "w")

X_size = tk.Entry(row31, width = 9)
X_size.grid(row = 0, column = 1, sticky = "w")
X_size.delete(0,"end")
X_size.insert(0,"0.5")

Ys_label = tk.Label(row31, text="Y: ")
Ys_label.grid(row = 0, column = 2, sticky = "w")

Y_size = tk.Entry(row31, width = 9)
Y_size.grid(row = 0, column = 3, sticky = "w")
Y_size.delete(0,"end")
Y_size.insert(0,"0.5")

Zs_label = tk.Label(row31, text="Z: ")
Zs_label.grid(row = 0, column = 4, sticky = "w")

Z_size = tk.Entry(row31, width = 9)
Z_size.grid(row = 0, column = 5, sticky = "w")
Z_size.delete(0,"end")
Z_size.insert(0,"0.5")

row41 = ttk.Frame(frame1)
row41.pack(side="top", fill="x")
Res_label = tk.Label(row41, text="Resolution (nm): ")
Res_label.grid(row = 0, column = 0, sticky = "w")

row51 = ttk.Frame(frame1)
row51.pack(side="top", fill="x")
Xr_label = tk.Label(row51, text="X: ")
Xr_label.grid(row = 0, column = 0, sticky = "w")


X_resol = tk.Entry(row51, width = 9)
X_resol.grid(row = 0, column = 1, sticky = "w")
X_resol.delete(0,"end")
X_resol.insert(0,"200")

Yr_label = tk.Label(row51, text="Y: ")
Yr_label.grid(row = 0, column = 2, sticky = "w")

Y_resol = tk.Entry(row51, width = 9)
Y_resol.grid(row = 0, column = 3, sticky = "w")
Y_resol.delete(0,"end")
Y_resol.insert(0,"200")

Zr_label = tk.Label(row51, text="Z: ")
Zr_label.grid(row = 0, column = 4, sticky = "w")

Z_resol = tk.Entry(row51, width = 9)
Z_resol.grid(row = 0, column = 5, sticky = "w")
Z_resol.delete(0,"end")
Z_resol.insert(0,"500")

row51 = ttk.Frame(frame1)
row51.pack(side="top", fill="x")
Time_label = tk.Label(row51, text="Time parameters: ")
Time_label.grid(row = 0, column = 0, sticky = "w")

row61 = ttk.Frame(frame1)
row61.pack(side="top", fill="x")
SimT_label = tk.Label(row61, text="Simulation time (seconds): ")
SimT_label.grid(row = 0, column = 0, sticky = "w")
Sim_time = tk.Entry(row61, width = 9)
Sim_time.grid(row = 0, column = 1, sticky = "w")
Sim_time.delete(0,"end")
Sim_time.insert(0,"10")

StepT_label = tk.Label(row61, text="Time step (milliseconds): ")
StepT_label.grid(row = 1, column = 0, sticky = "w")
Step_time = tk.Entry(row61, width = 9)
Step_time.grid(row = 1, column = 1, sticky = "w")
Step_time.delete(0,"end")
Step_time.insert(0,"0.001")

DwellT_label = tk.Label(row61, text="Dwell time (milliseconds): ")
DwellT_label.grid(row = 2, column = 0, sticky = "w")
Dwell_time = tk.Entry(row61, width = 9)
Dwell_time.grid(row = 2, column = 1, sticky = "w")
Dwell_time.delete(0,"end")
Dwell_time.insert(0,"0.01")

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
tabs.add(frame2, text = "Particles")

#------------------------------------------------------------------------------------------------
#------------------------------         Particles      ------------------------------------------
#------------------------------------------------------------------------------------------------
row12 = ttk.Frame(frame2)
row12.pack(side="top", fill="x")

Add_particle_Button = tk.Button(row12, text="Add", command=Add)
Add_particle_Button.pack(side = "left", anchor = "nw")

Edit_particle_Button = tk.Button(row12, text="Edit", command=Edit)
Edit_particle_Button.pack(side = "left", anchor = "nw")

Remove_particle_Button = tk.Button(row12, text="Remove", command=Remove)
Remove_particle_Button.pack(side = "left", anchor = "nw")

row22 = ttk.Frame(frame2)
row22.pack(side="top", fill="x")

particle_list = ttk.Treeview(row22, columns=(1,2,3,4,5), show = "headings")
particle_list.pack(side = "left", anchor = "nw")

particle_list.heading(1, text="Number")
particle_list.column(1, width = round(frame0_width/5))
particle_list.heading(2, text="Green units")
particle_list.column(2, width = round(frame0_width/5))
particle_list.heading(3, text="Red units")
particle_list.column(3, width = round(frame0_width/5))
particle_list.heading(4, text="Diff. coeff." + "\n" + "(um^2/s)")
particle_list.column(4, width = round(frame0_width/5))
particle_list.heading(5, text="Diff. coeff. in trap" + "\n" + "(um^2/s)")
particle_list.column(5, width = round(frame0_width/5))

particle_list.bind('<<TreeviewSelect>>', Select_row)

#ttk.Style().configure("Treeview", rowheight=50)

row32 = ttk.Frame(frame2)
row32.pack(side="top", fill="x")

Green_intensity_label = tk.Label(row32, text="Intensity of one green unit (a.u.): ")
Green_intensity_label.grid(row = 0, column = 0, sticky = "w")

Green_intensity_entry = tk.Entry(row32, width = 9)
Green_intensity_entry.grid(row = 0, column = 1, sticky = "w")
Green_intensity_entry.delete(0,"end")
Green_intensity_entry.insert(0,"1")

Red_intensity_label = tk.Label(row32, text="Intensity of one red unit (a.u.): ")
Red_intensity_label.grid(row = 1, column = 0, sticky = "w")

Red_intensity_entry = tk.Entry(row32, width = 9)
Red_intensity_entry.grid(row = 1, column = 1, sticky = "w")
Red_intensity_entry.delete(0,"end")
Red_intensity_entry.insert(0,"1")



#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
tabs.add(frame3, text = "Frames/plots")

#------------------------------------------------------------------------------------------------
#--------------------------------      Frames/plots    ------------------------------------------
#------------------------------------------------------------------------------------------------

plot_col = Plot_colors("#00ff00", "#ff00ff", "#00ff00", "#ff00ff", "#00ff00", "#ff00ff", "#ff8040", "#0000ff")

row13 = ttk.Frame(frame3)
row13.pack(side="top", fill="x")

Pcolors_label = tk.Label(row13, text="Particles: ")
Pcolors_label.grid(row = 0, column = 0, sticky = "w")

row23 = ttk.Frame(frame3)
row23.pack(side="top", fill="x")

PGreencolor_label = tk.Label(row23, text="Green particles: ")
PGreencolor_label.grid(row = 0, column = 0, sticky = "w")

PGreencolor_entry = tk.Entry(row23, width = 9)
PGreencolor_entry.grid(row = 0, column = 1, sticky = "w")
PGreencolor_entry.delete(0,"end")
PGreencolor_entry.insert(0,"#00ff00")
PGreencolor_entry.config(readonlybackground  = "#00ff00")
PGreencolor_entry.config(state = 'readonly')

PGreencolor_Button = tk.Button(row23, text="Choose color", command=PG_color)
PGreencolor_Button.grid(row = 0, column = 2, sticky = "w")

PRedcolor_label = tk.Label(row23, text="Red particles: ")
PRedcolor_label.grid(row = 1, column = 0, sticky = "w")

PRedcolor_entry = tk.Entry(row23, width = 9)
PRedcolor_entry.grid(row = 1, column = 1, sticky = "w")
PRedcolor_entry.delete(0,"end")
PRedcolor_entry.insert(0,"#ff00ff")
PRedcolor_entry.config(readonlybackground  = "#ff00ff")
PRedcolor_entry.config(state = 'readonly')


PRedcolor_Button = tk.Button(row23, text="Choose color", command=PR_color)
PRedcolor_Button.grid(row = 1, column = 2, sticky = "w")


Tcolors_label = tk.Label(row23, text="Traces: ")
Tcolors_label.grid(row = 2, column = 0, columnspan = 2, sticky = "w")

TGreencolor_label = tk.Label(row23, text="Green traces: ")
TGreencolor_label.grid(row = 3, column = 0, sticky = "w")

TGreencolor_entry = tk.Entry(row23, width = 9)
TGreencolor_entry.grid(row = 3, column = 1, sticky = "w")
TGreencolor_entry.delete(0,"end")
TGreencolor_entry.insert(0,"#00ff00")
TGreencolor_entry.config(readonlybackground  = "#00ff00")
TGreencolor_entry.config(state = 'readonly')

TGreencolor_Button = tk.Button(row23, text="Choose color", command=TG_color)
TGreencolor_Button.grid(row = 3, column = 2, sticky = "w")

TRedcolor_label = tk.Label(row23, text="Red traces: ")
TRedcolor_label.grid(row = 4, column = 0, sticky = "w")

TRedcolor_entry = tk.Entry(row23, width = 9)
TRedcolor_entry.grid(row = 4, column = 1, sticky = "w")
TRedcolor_entry.delete(0,"end")
TRedcolor_entry.insert(0,"#ff00ff")
TRedcolor_entry.config(readonlybackground  = "#ff00ff")
TRedcolor_entry.config(state = 'readonly')

TRedcolor_Button = tk.Button(row23, text="Choose color", command=TR_color)
TRedcolor_Button.grid(row = 4, column = 2, sticky = "w")




Ccolors_label = tk.Label(row23, text="Correlations: ")
Ccolors_label.grid(row = 5, column = 0, columnspan = 2, sticky = "w")

CGreencolor_label = tk.Label(row23, text="Green autocorr: ")
CGreencolor_label.grid(row = 6, column = 0, sticky = "w")

CGreencolor_entry = tk.Entry(row23, width = 9)
CGreencolor_entry.grid(row = 6, column = 1, sticky = "w")
CGreencolor_entry.delete(0,"end")
CGreencolor_entry.insert(0,"#00ff00")
CGreencolor_entry.config(readonlybackground  = "#00ff00")
CGreencolor_entry.config(state = 'readonly')

CGreencolor_Button = tk.Button(row23, text="Choose color", command=CG_color)
CGreencolor_Button.grid(row = 6, column = 2, sticky = "w")

CRedcolor_label = tk.Label(row23, text="Red autocorr: ")
CRedcolor_label.grid(row = 7, column = 0, sticky = "w")

CRedcolor_entry = tk.Entry(row23, width = 9)
CRedcolor_entry.grid(row = 7, column = 1, sticky = "w")
CRedcolor_entry.delete(0,"end")
CRedcolor_entry.insert(0,"#ff00ff")
CRedcolor_entry.config(readonlybackground  = "#ff00ff")
CRedcolor_entry.config(state = 'readonly')

CRedcolor_Button = tk.Button(row23, text="Choose color", command=CR_color)
CRedcolor_Button.grid(row = 7, column = 2, sticky = "w")




Cross_label = tk.Label(row23, text="Cross corr: ")
Cross_label.grid(row = 8, column = 0, sticky = "w")

Cross_entry = tk.Entry(row23, width = 9)
Cross_entry.grid(row = 8, column = 1, sticky = "w")
Cross_entry.delete(0,"end")
Cross_entry.insert(0,"#ff8040")
Cross_entry.config(readonlybackground  = "#ff8040")
Cross_entry.config(state = 'readonly')

Cross_Button = tk.Button(row23, text="Choose color", command=CC_color)
Cross_Button.grid(row = 8, column = 2, sticky = "w")

Focalspot_label = tk.Label(row23, text="Focal spot: ")
Focalspot_label.grid(row = 9, column = 0, columnspan = 2, sticky = "w")

Focal_label = tk.Label(row23, text="Color: ")
Focal_label.grid(row = 10, column = 0, sticky = "w")

Focal_entry = tk.Entry(row23, width = 9)
Focal_entry.grid(row = 10, column = 1, sticky = "w")
Focal_entry.delete(0,"end")
Focal_entry.insert(0,"#0000ff")
Focal_entry.config(readonlybackground  = "#0000ff")
Focal_entry.config(state = 'readonly')

Focal_Button = tk.Button(row23, text="Choose color", command=FS_color)
Focal_Button.grid(row = 10, column = 2, sticky = "w")

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
tabs.add(frame4, text = "Traps")

#------------------------------------------------------------------------------------------------
#-----------------------------------      Traps      --------------------------------------------
#------------------------------------------------------------------------------------------------

var1 = tk.IntVar()
Check_trap_button=tk.Checkbutton(frame4, text="Traps", variable=var1, command=Check_trap)
Check_trap_button.grid(row = 0, column = 0, columnspan=2, sticky = "w")


Traps_number_label = tk.Label(frame4, text="Number: ")
Traps_number_label.grid(row = 1, column = 0, sticky = "w")

Traps_number_entry = tk.Entry(frame4, width = 9)
Traps_number_entry.grid(row = 1, column = 1, sticky = "w")
Traps_number_entry.delete(0,"end")
Traps_number_entry.insert(0,"9")

Traps_radii_label = tk.Label(frame4, text="Radii: ")
Traps_radii_label.grid(row = 2, column = 0, sticky = "w")

Traps_radii_entry = tk.Entry(frame4, width = 9)
Traps_radii_entry.grid(row = 2, column = 1, sticky = "w")
Traps_radii_entry.delete(0,"end")
Traps_radii_entry.insert(0,"0.03")

var2 = tk.IntVar()
Check_trap_diff_button=tk.Checkbutton(frame4, text="Traps diffuse", variable=var2, command=Check_trap_diff)
Check_trap_diff_button.grid(row = 3, column = 0, columnspan=2, sticky = "w")

Traps_diff_label = tk.Label(frame4, text="Diffusion: ")
Traps_diff_label.grid(row = 4, column = 0, sticky = "w")

Traps_diff_entry = tk.Entry(frame4, width = 9)
Traps_diff_entry.grid(row = 4, column = 1, sticky = "w")
Traps_diff_entry.delete(0,"end")
Traps_diff_entry.insert(0,"0.01")


Traps_number_entry.config(state = "disabled")
Traps_diff_entry.config(state = "disabled")
Traps_radii_entry.config(state = "disabled")
Check_trap_diff_button.config(state = "disabled")
Check_trap_button.config(state = "disabled")


#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
tabs.add(frame5, text = "K_on/K_off")


tabs_number = 4;

tabs.pack(side="top", fill="x")

frame03 = tk.Frame(frame0)
frame03.pack(side="top", fill="x")

Output_directory_label = tk.Label(frame03, text="Output directory: ")
Output_directory_label.pack(side = "left", anchor = "nw")


Output_directory = tk.Entry(frame03, width = 30)
Output_directory.pack(side = "left", anchor = "nw")

Output_directory_Button = tk.Button(frame03, text="Browse", command=Import)
Output_directory_Button.pack(side = "left", anchor = "nw")

frame02 = tk.Frame(frame0)
frame02.pack(side="top", fill="x")

Preview_Button = tk.Button(frame02, text="Preview", command=Import)
Preview_Button.pack(side = "left", anchor = "nw")

Run_Button = tk.Button(frame02, text="Run", command=Run)
Run_Button.pack(side = "left", anchor = "nw")


frame00 = tk.LabelFrame(root, text = "Preview")
frame00.pack(side = "left", anchor = "nw", expand =1, fill=tk.BOTH)
frame00.config(bd=0, width = round(win_width*3/4), height = win_height)
frame00.grid_propagate(0)


root.mainloop()