import matplotlib.ticker
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import moviepy.editor as mp
import os 

samplerate_59, data_59 = wavfile.read('./audio/59.wav')
samplerate_SC, data_SC = wavfile.read('./audio/Single.wav')

data_59 = data_59[:, 0] # accidentally exported as stereo....   
data_SC = data_SC[:, 0]

window_size = 2**13  # adjust window size for better frequency resolution
shift_amount_per_frame_59 = int((1/30) * samplerate_59)  # shift for 30fps
shift_amount_per_frame_SC = int((1/30) * samplerate_SC)  # shift for 30fps

def get_buffer_fft(data, buffer_index, window_size, sampling_rate):
    buffer = data[buffer_index:(buffer_index + window_size)]
    fft = np.fft.fft(buffer)/window_size
    frequencies = np.fft.fftfreq(window_size, 1/sampling_rate)
    return np.abs(fft)[:window_size], np.abs(frequencies)[:window_size]

fig, ax = plt.subplots()
line_59, = ax.plot([], [], label='59.wav')
line_SC, = ax.plot([], [], label='Single.wav')
xTickMarks = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
ax.set_xscale('symlog')
ax.xaxis.set_major_locator(matplotlib.ticker.SymmetricalLogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, linthresh=2))
ax.set_xlim(0, samplerate_59 // 2)
ax.set_xticks(xTickMarks, labels=[f'{i/1000}k' if i >= 1000 else str(i) for i in xTickMarks])
ax.set_ylim(0, 1000)  
ax.set_title("FFT of Audio Signals")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.legend()

def animate(frame):
    buffer_index_59 = frame * shift_amount_per_frame_59
    buffer_index_SC = frame * shift_amount_per_frame_SC
    
    fft_59, frequencies_59 = get_buffer_fft(data_59, buffer_index_59, window_size, samplerate_59)
    fft_SC, frequencies_SC = get_buffer_fft(data_SC, buffer_index_SC, window_size, samplerate_SC)
    
    line_59.set_data(frequencies_59[:window_size//2], fft_59[:window_size//2])
    line_SC.set_data(frequencies_SC[:window_size//2], fft_SC[:window_size//2])
    
    
    return line_59


anim_length = min(len(data_59) // shift_amount_per_frame_59, len(data_SC) // shift_amount_per_frame_SC) - 10

anim = animation.FuncAnimation(fig, animate, frames=anim_length, interval=33)

# plt.show()
if (os.path.isfile("./Output.mp4")):
    os.remove("./Output.mp4")

anim.save('./videoOut.mp4')

videoOut = mp.VideoFileClip('./videoOut.mp4')
audioSC = mp.AudioFileClip('./audio/Single.wav')
composited = videoOut.set_audio(audioSC)
composited.write_videofile("./Output.mp4")

os.remove('./videoOut.mp4')
