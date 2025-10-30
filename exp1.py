import numpy as np
import matplotlib.pyplot as plt
# Define temperature and fan speed ranges
temperature = np.arange(0, 41, 1)
fan = np.arange(0, 101, 1)
# --- Fuzzy membership functions for Temperature ---
def cold(x):
    return np.clip((20 - x) / 20, 0, 1)
def warm(x):
    return np.clip(np.minimum((x - 15) / 10, (35 - x) / 10), 0, 1)
def hot(x):
    return np.clip((x - 30) / 10, 0, 1)
# --- Fuzzy membership functions for Fan Speed ---
def slow(x):
    return np.clip((50 - x) / 50, 0, 1)
def medium(x):
    return np.clip(np.minimum((x - 25) / 25, (75 - x) / 25), 0, 1)
def fast(x):
    return np.clip((x - 50) / 50, 0, 1)
# --- Fuzzy inference rule (simple approximation) ---
def fan_speed(temp):
    c = cold(temp)
    w = warm(temp)
    h = hot(temp)
    speed = (c * 20 + w * 50 + h * 90) / (c + w + h)
    return speed
# Example: user input
user_temp = 28
speed = fan_speed(user_temp)
print(f"\nTemperature input: {user_temp}" )

print(f"Calculated fan speed: {speed:.2f} ")
# --- Plot Temperature Membership Functions ---
plt.figure(figsize=(7, 4))
plt.plot(temperature, [cold(x) for x in temperature], label='Cold')
plt.plot(temperature, [warm(x) for x in temperature], label='Warm')
plt.plot(temperature, [hot(x) for x in temperature], label='Hot')
plt.xlabel('Temperature (°C)')
plt.ylabel('Membership Degree')
plt.title('Temperature Membership Functions')
plt.legend()
plt.grid(True)
plt.show()
# --- Plot Fan Speed Membership Functions ---
plt.figure(figsize=(7, 4))
plt.plot(fan, [slow(x) for x in fan], label='Slow')
plt.plot(fan, [medium(x) for x in fan], label='Medium')
plt.plot(fan, [fast(x) for x in fan], label='Fast')
plt.xlabel('Fan Speed (%)')
plt.ylabel('Membership Degree')
plt.title('Fan Speed Membership Functions')
plt.legend()
plt.grid(True)
plt.show()