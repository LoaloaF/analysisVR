import numpy as np
import matplotlib.pyplot as plt


# amp = 25

# # 50 us * 20kHz = 1000 samples
# sampling_rate = 20000  # 20 kHz
# f = 500
# # duration = 0.05  # 50 microseconds

# # 20 = 20000/1000

# # Create a time array
# t = np.linspace(0,1, int(sampling_rate/f))

# # Create a sine wave with a frequency of 1 kHz
# # frequency = 1000  # 1 kHz
# sine_wave = amp * np.sin(t*2*np.pi) +512

# # Print the sine wave array
# print(sine_wave)
# plt.plot(t, sine_wave)
# plt.show()



# READ THIS

d = np.load("sine_wave.npy")

plt.plot(d)
plt.show()