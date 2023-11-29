import wfdb
import matplotlib.pyplot as plt


record = wfdb.rdheader('fetal_PCG_p15_GW_36')  
signals, fields = wfdb.rdsamp('fetal_PCG_p15_GW_36')


base_time = fields['base_time'] if fields['base_time'] is not None else 0


new_fs = fields['fs'] * 50  
time = new_fs * (signals[:, 0] - base_time) / fields['fs']


fig = plt.figure()

num_channels = len(signals[0])
for i in range(num_channels):
    plt.plot(time, signals[:, i], label=f'Channel {i + 1}')

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.title('Biomedical Signal')
plt.legend()
plt.show()


