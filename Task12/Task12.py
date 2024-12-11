import numpy as np
import matplotlib.pyplot as plt

# Параметры сигнала
a0 = 1
a1 = 0.001  # Амплитуда второго сигнала
omega0 = 5.1
omega1 = 25.5
T = 2 * np.pi
fs = 100000  # Частота дискретизации
N = int(T * fs)  # Количество точек увеличено
t = np.linspace(0, T, N, endpoint=False)  # Временной вектор

f_t = a0 * np.sin(omega0 * t) + a1 * np.sin(omega1 * t)

# Окна
rect_window = np.ones(N)
hann_window = np.hanning(N)

# Применение окон
f_t_rect = f_t * rect_window
f_t_hann = f_t * hann_window

# FFT
freqs = np.fft.fftfreq(N, 1 / N)[:N // 2]
fft_rect = np.fft.fft(f_t_rect)[:N // 2]
fft_hann = np.fft.fft(f_t_hann)[:N // 2]

power_rect = np.abs(fft_rect) ** 2
power_hann = np.abs(fft_hann) ** 2

# Логарифмический масштаб (децибелы)
power_rect_db = 10 * np.log10(power_rect / np.max(power_rect))
power_hann_db = 10 * np.log10(power_hann / np.max(power_hann))

# Построение графиков
plt.figure(figsize=(12, 6))

# Прямоугольное окно
plt.subplot(1, 2, 1)
plt.plot(freqs, power_rect_db, label="Rectangular Window", color="blue")
plt.xlim(0, 60)
plt.ylim(-100, 0)  # Ограничение по оси Y для улучшения видимости
plt.title("Power Spectrum (Rectangular Window)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.grid()
plt.legend()

# Окно Ханна
plt.subplot(1, 2, 2)
plt.plot(freqs, power_hann_db, label="Hanning Window", color="orange")
plt.xlim(0, 60)
plt.ylim(-100, 0)  # Ограничение по оси Y для согласованности
plt.title("Power Spectrum (Hanning Window)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dB)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
