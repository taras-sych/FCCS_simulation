import numpy as np
import matplotlib.pyplot as plt
import Correlation as corr_py
from scipy.stats import norm
import pandas as pd

# Simulation parameters
N = 500000 # Number of time points
Nmol = 3 # Number of particles
D = 40e-14 # Diffusion coefficient in m^2/s
tau = 10e-6 # Time interval between measurements in s
w = np.array([200e-9, 200e-9, 500e-9]) # Focal volume dimensions in m

# Initialize particle positions
x1 = np.zeros((N,Nmol))
x1[0,:] = np.random.rand(Nmol)*w[0]
y1 = np.zeros((N,Nmol))
y1[0,:] = np.random.rand(Nmol)*w[1]

# Simulate particle diffusion
for i in range(1,N):
    dx = np.sqrt(2*D*tau)*np.random.randn(Nmol)
    dy = np.sqrt(2*D*tau)*np.random.randn(Nmol)
    x1[i,:] = x1[i-1,:] + dx
    y1[i,:] = y1[i-1,:] + dy
    # Apply periodic boundary conditions
    x1[i,:] = np.mod(x1[i,:], w[0])
    y1[i,:] = np.mod(y1[i,:], w[1])

# Calculate intensity fluctuations
G1 = np.zeros(N)
for i in range(Nmol):
    for j in range(Nmol):
        if j != i:
            r = np.sqrt((x1[:,i]-x1[:,j])**2 + (y1[:,i]-y1[:,j])**2)
            G1 += np.exp(-4*r**2/(w[0]**2 + w[1]**2)) / (np.pi**(3/2) * w[0] * w[1] * r)
G1 /= (Nmol*(Nmol-1))

Nmol = 2
# Initialize particle positions
x2 = np.zeros((N,Nmol))
x2[0,:] = np.random.rand(Nmol)*w[0]
y2 = np.zeros((N,Nmol))
y2[0,:] = np.random.rand(Nmol)*w[1]

# Simulate particle diffusion
for i in range(1,N):
    dx = np.sqrt(2*D*tau)*np.random.randn(Nmol)
    dy = np.sqrt(2*D*tau)*np.random.randn(Nmol)
    x2[i,:] = x2[i-1,:] + dx
    y2[i,:] = y2[i-1,:] + dy
    # Apply periodic boundary conditions
    x2[i,:] = np.mod(x2[i,:], w[0])
    y2[i,:] = np.mod(y2[i,:], w[1])

# Calculate intensity fluctuations
G2 = np.zeros(N)
for i in range(Nmol):
    for j in range(Nmol):
        if j != i:
            r = np.sqrt((x2[:,i]-x2[:,j])**2 + (y2[:,i]-y2[:,j])**2)
            G2 += np.exp(-4*r**2/(w[0]**2 + w[1]**2)) / (np.pi**(3/2) * w[0] * w[1] * r)
G2 /= (Nmol*(Nmol-1))

t = np.arange(N)*tau

i = 0
G1_save = []
G2_save = []
t_save = []
while i<len(G1):
    temp = sum(G1[i:i+100])/100
    G1_save.append(temp)
    temp = sum(G2[i:i+100])/100
    G2_save.append(temp)
    temp = t[i]
    t_save.append(temp)
    i += 100


df1 = pd.DataFrame({'Time': t, 'Channel 1': G1, 'Channel 2': G2})

plt.plot(t, G1)
plt.plot(t, G2)
plt.xlabel('Time (s)')
plt.ylabel('Intensity Autocorrelation')
plt.show()


timestep = t[1] - t[0]
x1, y1 = corr_py.correlate_full (timestep, G1, G1)
x2, y2 = corr_py.correlate_full (timestep, G2, G2)
x3, y3 = corr_py.correlate_full (timestep, G1, G2)

df2 = pd.DataFrame({'Time': x1, 'AutoCorr 1': y1, 'Time': x2, 'AutoCorr 2': y2, 'Time': x3, 'CrossCorr 1': y3})

plt.plot(x1, y1, label='Trace 1')
plt.plot(x2, y2, label='Trace 2')
plt.plot(x3, y3, label='Cross-corr')
plt.xscale('log')
plt.xlabel('Delay time')
plt.ylabel('G(tau)')
plt.title('FCCS')
plt.legend()
plt.show()

# Create the Gaussian peak with intensity 
data = np.zeros(len(G1))
magnitude = np.max(G1)*0.008
gaussian = norm.pdf(t, loc=2, scale=0.0025) * magnitude

# Add the Gaussian peak to the list
data += gaussian

G1 = np.array(G1) + np.array(data)


data = np.zeros(len(G1))
magnitude = np.max(G1)*0.02
gaussian = norm.pdf(t, loc=2, scale=0.0025) * magnitude

# Add the Gaussian peak to the list
data += gaussian

G2 = np.array(G2) + np.array(data)

i = 0
G1_save = []
G2_save = []
t_save = []
while i<len(G1):
    temp = sum(G1[i:i+100])/100
    G1_save.append(temp)
    temp = sum(G2[i:i+100])/100
    G2_save.append(temp)
    temp = t[i]
    t_save.append(temp)
    i += 100

df3 = pd.DataFrame({'Time': t, 'Channel 1': G1, 'Channel 2': G2})


# Plot results

plt.plot(t, G1)
plt.plot(t, G2)
plt.xlabel('Time (s)')
plt.ylabel('Intensity Autocorrelation')
plt.show()


timestep = t[1] - t[0]
x1, y1 = corr_py.correlate_full (timestep, G1, G1)
x2, y2 = corr_py.correlate_full (timestep, G2, G2)
x3, y3 = corr_py.correlate_full (timestep, G1, G2)

df4 = pd.DataFrame({'Time': x1, 'AutoCorr 1': y1, 'Time': x2, 'AutoCorr 2': y2, 'Time': x3, 'CrossCorr 1': y3})

plt.plot(x1, y1, label='Trace 1')
plt.plot(x2, y2, label='Trace 2')
plt.plot(x3, y3, label='Cross-corr')
plt.xscale('log')
plt.xlabel('Delay time')
plt.ylabel('G(tau)')
plt.title('FCCS')
plt.legend()
plt.show()


filename1 = "C:\\Users\\taras.sych\\OneDrive - Karolinska Institutet\\Science\\Papers\\2021 - Fluctuometry paper\\Revision\\Supplementary figure - High cross correlaiton\\Simulate FCS\\Summary.xlsx"



# Create a Pandas Excel writer using the XlsxWriter engine
writer = pd.ExcelWriter(filename1, engine='xlsxwriter')

# Write the DataFrame to a specific sheet
sheet_name = 'Fluctuations no spike'
df1.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

sheet_name = 'FCCS no spike'
df2.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

# Write the DataFrame to a specific sheet
sheet_name = 'Fluctuations spike'
df3.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

sheet_name = 'FCCS spike'
df4.to_excel(writer, sheet_name=sheet_name, index=False, header=False)


# Save the Excel file
writer.save()