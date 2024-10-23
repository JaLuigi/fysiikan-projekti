import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from streamlit_folium import st_folium
from geopy.distance import geodesic

# Ladataan GPS- ja kiihtyvyysdata
url_gps = "https://raw.githubusercontent.com/JaLuigi/fysiikan-projekti/refs/heads/main/Location.csv"
url_acc = "https://raw.githubusercontent.com/JaLuigi/fysiikan-projekti/refs/heads/main/Linear%20Accelerometer.csv"

df_gps = pd.read_csv(url_gps)
df_acc = pd.read_csv(url_acc)

st.title("Päivän liikunta")

# Suodatetun kiihtyvyysdatan laskeminen
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Suodatusasetukset
fs = 50  # Näytteenottotaajuus
cutoff = 3  # Suodatinraja

# Suodatetaan Z-koordinaatti
df_acc['Z_filtered'] = butter_lowpass_filter(df_acc['Z (m/s^2)'], cutoff, fs)

# Lasketaan kiihtyvyyden suuruus ja suodatetaan se
acc_magnitude = np.sqrt(df_acc['X (m/s^2)']**2 + df_acc['Y (m/s^2)']**2 + df_acc['Z (m/s^2)']**2)
filtered_acc = butter_lowpass_filter(acc_magnitude, cutoff=2.5, fs=fs)

# Lasketaan askelmäärä suodatetusta kiihtyvyysdatasta
threshold = np.mean(filtered_acc) + 0.1 * np.std(filtered_acc)
steps_filtered = np.sum((filtered_acc[1:] > threshold) & (filtered_acc[:-1] < threshold))

# Fourier-analyysi askelmäärän laskemiseksi
n = len(acc_magnitude)
yf = fft(acc_magnitude)
xf = np.linspace(0.0, fs/2, n//2)

# Lasketaan dominantti taajuus Z-koordinaatista
acceleration_z = df_acc['Z (m/s^2)'].dropna().values
n_z = len(acceleration_z)
yf_z = fft(acceleration_z)
xf_z = np.linspace(0.0, fs/2, n_z//2)

dominant_freq = xf_z[np.argmax(2.0/n_z * np.abs(yf_z[:n_z//2]))]
steps_fourier = dominant_freq * (df_acc['Time (s)'].iloc[-1] - df_acc['Time (s)'].iloc[0])

# GPS-datasta kuljetun matkan ja keskinopeuden laskeminen
def calculate_distance(gps_data):
    total_distance = 0.0
    for i in range(1, len(gps_data)):
        coord1 = (gps_data['Latitude (°)'].iloc[i-1], gps_data['Longitude (°)'].iloc[i-1])
        coord2 = (gps_data['Latitude (°)'].iloc[i], gps_data['Longitude (°)'].iloc[i])
        total_distance += geodesic(coord1, coord2).kilometers
    return total_distance

distance_km = calculate_distance(df_gps)
time_elapsed = df_gps['Time (s)'].max() - df_gps['Time (s)'].min()
average_speed = distance_km / (time_elapsed / 3600)

# Askelpituuden laskeminen
step_length = (distance_km * 1000) / steps_filtered if steps_filtered > 0 else 0

# Kuvaaja suodatetusta kiihtyvyysdatasta
st.write("Suodatettu kiihtyvyysdata")
plt.figure(figsize=(10, 6))
time_data = df_acc['Time (s)']  # Aseta aika muuttuja
plt.plot(time_data, filtered_acc, label="Suodatettu kiihtyvyys")
plt.xlabel("Aika (s)")
plt.ylabel("Kiihtyvyys (m/s^2)")
plt.legend()
st.pyplot(plt)

# Kuvaaja tehospektritiheydestä Fourier-analyysin perusteella
st.write("Kiihtyvyysdatan tehospektritiheys")
plt.figure(figsize=(10, 6))
plt.plot(xf_z, 2.0/n_z * np.abs(yf_z[:n_z//2]), label="Tehospektritiheys")
plt.xlabel("Taajuus (Hz)")
plt.ylabel("Teho")
plt.legend()
st.pyplot(plt)

# Kartta reitistä
st.title("Reitti kartalla")
start_lat = df_gps['Latitude (°)'].mean()
start_long = df_gps['Longitude (°)'].mean()
map = folium.Map(location=[start_lat, start_long], zoom_start=14)

folium.PolyLine(df_gps[['Latitude (°)', 'Longitude (°)']].values, color='blue', weight=3.5, opacity=1).add_to(map)
st_map = st_folium(map, width=900, height=650)

# Tulostetaan laskelmat
st.write(f"Askeleet suodatetusta kiihtyvyysdatasta: {steps_filtered}")
st.write(f"Askeleet Fourier-analyysistä: {int(steps_fourier)}")
st.write(f"Kuljettu matka: {distance_km:.2f} km")
st.write(f"Keskinopeus: {average_speed:.2f} km/h")
st.write(f"Askelpituus: {step_length:.2f} m")
