import numpy as np
import matplotlib.pyplot as plt


class AmplificationEstimation:
    """Class to estimate the amplification within a crystal"""

    def __init__(
            self,
            power_signal: float = 1,
            power_pump: float = 50.0,
            absorption_signal: float = 0.25,
            absorption_pump: float = 0.5,
            wavelength_signal: float = 635.0,
            wavelength_pump: float = 408.0,
            subdivisions: int = 100,
            max_length: float = 20.0,
            max_step: float = 0.5,
            mode_size_z: float = 0.7,
            mode_size_y: float = 0.7,
            power_array: list = [30, 60],
            min_power: float = 0,
            seed_bandwidth: float = 0.2,
    ):
        """
        Initialization of the class instance

        Parameters
        ----------
        power_signal: float
            initial signal power in mW
        power_pump: float
            initial pump power in mW
        absorption_signal: float
            absorption of signal
        absorption_pump: float
            absorption of pump
        wavelength_signal:
            wavelength of signal in nm
        wavelength_pump: float
            wavelength of pump in nm
        subdivisions: int
            number of subdivisions of crystal length
        max_length: float
            maximal length of the crystal in mm
        max_step: float
            step size of the main division
        mode_size_z: float
            mode size in z direction in um
        mode_size_y: float
            mode size in y direction in um
        power_array: list
            contains the input power which should compared within the plot of the power
            after crystal
        min_power
            miminum possible pump power to stop amplification process, now works only for 0, should be adjusted
        """
        self.power_signal = power_signal
        self.power_pump = power_pump
        self.absorption_signal = absorption_signal
        self.absorption_pump = absorption_pump
        self.wavelength_signal = wavelength_signal
        self.wavelength_pump = wavelength_pump
        self.max_length = max_length
        self.max_step = max_step
        self.max_length_array = np.linspace(
            0, self.max_length, num=int(self.max_length / self.max_step) + 1
        )
        self.mode_size_z = mode_size_z
        self.mode_size_y = mode_size_y
        self.intensity_factor = 2.6 * 3.6 / (self.mode_size_z * self.mode_size_y)
        self.gain_factor = 0.00336 * np.sqrt(self.intensity_factor)
        self.power_array = power_array
        self.subdivisions = subdivisions
        self.min_power = min_power
        self.seed_bandwidth = seed_bandwidth

    def calculate_amplification(self, gain_all, length, power_overlap):
        """
        Calculating the amplification within the crystal for given crystal length

        Parameters
        ----------
        gain_all: float
            sum of the gain per crystal length
        length: float
            actual crystal length

        Returns
        -------
        power of the signal at specific length
        """
        return (
                power_overlap
                * np.cosh(gain_all) ** 2
                * 10 ** (-self.absorption_signal * length / 100)
        )

    def calculate_gain(self, power_pump_j, step):
        """
        Calculate the gain at each step

        Parameters
        ----------
        power_pump_j: float
            pump power at specific crystal length
        step:
            subdivision of crystal / step size

        Returns
        -------
        gain per step
        """
        if power_pump_j < self.min_power:
            return self.min_power
        else:
            return np.sqrt(power_pump_j) * self.gain_factor * step

    def calculate_power_pump_j(
            self, length, step, gain_all, gain_less, power_pump_j_before, power_overlap
    ):
        """
        Calculate pump power at specific crystal length

        Parameters
        ----------
        length: float
            current length
        step: float
            step size / subdivision of crystal
        gain_all: float
            sum of all gain
        gain_less: float
            sum of all gain without last value
        power_pump_j_before: float
            pump power before this calculation

        Returns
        -------
        pump power after specific crystal length
        """
        delta_nl = (
                -(self.wavelength_signal / self.wavelength_pump)
                * power_overlap
                * 10 ** (-(self.absorption_signal * length - step) / 100)
                * (np.sinh(gain_all) ** 2 - np.sinh(gain_less) ** 2)
        )
        delta_lin = 10 ** (-self.absorption_pump * step / 100) * power_pump_j_before
        if power_pump_j_before <= self.min_power:
            return self.min_power
        else:
            return delta_nl + delta_lin

    def calculate_power_after_crystal(self):
        """
        Calculate power of signal after crystal

        Returns
        -------
        power after crystal
        """
        pump_power = []
        amplification_list = []
        total_signal_power = []
        for length in self.max_length_array:
            if length == 0:
                bandwidth = 10 * self.seed_bandwidth
            else:
                bandwidth = 3.8 / length

            seed_overlap = min(bandwidth / self.seed_bandwidth, 1)
            power_overlap = seed_overlap * self.power_signal
            pump_power_sublist = []
            amplification_sublist = []
            gain_all_list = [0]
            sub_length_array = np.linspace(0, length, num=self.subdivisions + 1)
            sub_step = length / (self.subdivisions + 1)
            for sub_length in sub_length_array:
                if sub_length == 0:
                    power_pump_j = self.power_pump
                    gain_all_list.append(0)
                    amplification_sublist.append(power_overlap)
                    pump_power_sublist.append(power_pump_j)
                else:
                    if power_pump_j < self.min_power:
                        power_pump_j = 0
                    elif power_pump_j > self.power_pump:
                        power_pump_j = self.power_pump

                    gain_all_list.append(
                        self.calculate_gain(
                            power_pump_j=self.calculate_power_pump_j(
                                length=sub_length,
                                step=sub_step,
                                gain_all=sum(gain_all_list),
                                gain_less=sum(gain_all_list[:-1]),
                                power_pump_j_before=power_pump_j,
                                power_overlap=power_overlap,
                            ),
                            step=sub_step,
                        )
                    )
                    power_pump_j = self.calculate_power_pump_j(
                        length=sub_length,
                        step=sub_step,
                        gain_all=sum(gain_all_list),
                        gain_less=sum(gain_all_list[:-1]),
                        power_pump_j_before=power_pump_j,
                        power_overlap=power_overlap,
                    )
                    pump_power_sublist.append(power_pump_j)
                    amp = self.calculate_amplification(
                        gain_all=sum(gain_all_list),
                        length=sub_length,
                        power_overlap=power_overlap,
                    )
                    amplification_sublist.append(amp)
            amp = amplification_sublist[-1]
            pump_power.append(pump_power_sublist[-1])
            amplification_list.append(amp)
            total_signal_power.append(
                amp
                + (1 - seed_overlap) * 10 ** (-self.absorption_signal * length / 100)
            )
        return amplification_list, total_signal_power, pump_power

    def plot_power_signal(self):
        """
        Plot power after crystal for different initial pump powers and create figure.
        """
        fig, ax = plt.subplots()
        for power in self.power_array:
            self.power_pump = power
            power_after = self.calculate_power_after_crystal()[0]
            ax.set_xlabel("Length of crystal [mm]")
            ax.set_ylabel("Signal power [mW]")
            ax.set_xlim(0, self.max_length)
            ax.plot(
                self.max_length_array,
                power_after,
                label=f"{power} mW",
            )
            with open(f"signal_power_{power}.txt", "w") as f:
                f.write("length [mm]\t signal_power [mW]\n")
                for counter, length in enumerate(self.max_length_array):
                    f.write(f"{length}\t {power_after[counter]}\n")
        ax.legend()
        fig.savefig("Power_signal_with_bandwidth.pdf")

    def plot_total(self):
        fig, ax = plt.subplots()
        for power in self.power_array:
            self.power_pump = power
            power_after = self.calculate_power_after_crystal()[1]
            ax.set_xlabel("Length of crystal [mm]")
            ax.set_ylabel("Total power [mW]")
            ax.set_xlim(0, self.max_length)
            ax.plot(
                self.max_length_array,
                power_after,
                label=f"{power} mW",
            )
            with open(f"tot_power_{power}.txt", "w") as f:
                f.write("length [mm]\t tot_power [mW]\n")
                for counter, length in enumerate(self.max_length_array):
                    f.write(f"{length}\t {power_after[counter]}\n")
        ax.legend()
        fig.savefig("Power_total_with_bandwidth.pdf")

    def plot_pump_power(self):
        fig, ax = plt.subplots()
        for power in self.power_array:
            self.power_pump = power
            power_after = self.calculate_power_after_crystal()[2]
            ax.set_xlabel("Length of crystal [mm]")
            ax.set_ylabel("Pump power [mW]")
            ax.set_xlim(0, self.max_length)
            ax.plot(
                self.max_length_array,
                power_after,
                label=f"{power} mW",
            )
            with open(f"pump_power_{power}.txt", "w") as f:
                f.write("length [mm]\t pump_power [mW]\n")
                for counter, length in enumerate(self.max_length_array):
                    f.write(f"{length}\t {power_after[counter]}\n")
        ax.legend()
        fig.savefig("Power_pump_with_bandwidth.pdf")


get_amplification = AmplificationEstimation()
get_amplification.plot_power_signal()
get_amplification.plot_total()
get_amplification.plot_pump_power()
