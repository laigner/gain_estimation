import numpy as np
import matplotlib.pyplot as plt


class GainEstimation:

    def __call__(self, losses_pump: float, losses_sled: float, mode_size_z: float,
                 mode_size_y: float):
        self.losses_pump = losses_pump
        self.losses_sled = losses_sled
        self.mode_size_z = mode_size_z
        self.mode_size_y = mode_size_y
        self.pump_intensity = 0.1 / (self.mode_size_z * self.mode_size_y)
        self.intensity_factor = 2.6 * 3.6 / (self.mode_size_z * self.mode_size_y)
        self.gain_inside = 0.00336 * np.sqrt(self.intensity_factor)

    def calculate_pump_power_per_mode(self, length_min: float, length_max: float):
        length_list = [x for x in range(length_min, length_max)]
