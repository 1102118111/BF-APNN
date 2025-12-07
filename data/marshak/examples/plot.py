import numpy as np
import matplotlib.pyplot as plt

x,y,z = np.split(np.loadtxt('result500_0'),3,1)
#plt.plot(x,y**0.25,'r')
plt.plot(x,z,'g')
for ii in range(5):
    x,y,z = np.split(np.loadtxt('result500_%.1f'%(0.1*ii+0.1)),3,1)
    #plt.plot(x,y**0.25,'r')
    plt.plot(x,z,'g')
plt.show()


# x,y,z = np.split(np.loadtxt('result500'),3,1)
# plt.plot(x,y**0.25,'r')
# plt.plot(x,z,'g')
# #plt.show()
#
# x,y,z = np.split(np.loadtxt('result500_0.5'),3,1)
# plt.plot(x,y**0.25,'r')
# plt.plot(x,z,'g')
# plt.show()


# x,y,z = np.split(np.loadtxt('oldoutput/result2430'),3,1)
# plt.plot(x,y**0.25,'r')
# plt.plot(x,z,'g')
# plt.show()