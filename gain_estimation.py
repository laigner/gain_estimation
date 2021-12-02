import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GainEstimation:

    def __init__(self, losses_pump: float = 1, losses_sled: float = 0.5, mode_size_z:
    float = 0.7, mode_size_y: float = 0.7, length_min: float = 0.001, length_max:
    float = 50, number_of_modes: int = 1, initial_pump_power: float = 0.,
                 power_overlap_factor: float = 1, amp_bandwidth_factor: float = 3.2,
                 bandwidth_overlap_sled_factor: float = 5.7):
        """Initialize class object.
        
        Initialize the object e.g. test_object = GainEstimation(*args). If called 
        without arguments, default values will be used.
        
        Parameters
        ----------
        losses_pump: float
            loss of the pump in db/cm, default is 1 db/cm
        losses_sled: float
            loss of the SLED in db/cm, default is 0.5 db/cm
        mode_size_z: float
            size of the mode in z direction in um, default is 0.7 um
        mode_size_y: float
            size of the mode in y direction in um, default is 0.7 um
        length_min: float
            minimum length of the crystal in mm, default is 0.001 mm 
        length_max: float
            maximum length of crystal in mm, default is 50 mm
        number_of_modes: int
            number of modes, default is 1
        initial_pump_power: float
            initial pump power in mW, default is 0.1 mW
        power_overlap_factor: float
            percentage of power overlap, default is 1 (100%)
        amp_bandwidth_factor: float
            default is 3.2
        bandwidth_overlap_sled_factor: float
            default is 5.7
        """
        self.losses_pump = losses_pump
        self.losses_sled = losses_sled
        self.mode_size_z = mode_size_z
        self.mode_size_y = mode_size_y
        self.initial_pump_power = initial_pump_power
        self.pump_intensity = self.initial_pump_power / (self.mode_size_z *
                                                         self.mode_size_y)
        self.intensity_factor = 2.6 * 3.6 / (self.mode_size_z * self.mode_size_y)
        self.gain_inside = 0.00336 * np.sqrt(self.intensity_factor)
        self.length_min = length_min
        self.length_max = length_max
        self.number_of_modes = number_of_modes
        self.power_overlap_factor = power_overlap_factor
        self.amp_bandwidth_factor = amp_bandwidth_factor
        self.bandwidth_overlap_sled_factor = bandwidth_overlap_sled_factor
        self.length_array = np.arange(length_min, length_max)
        self.amp_bandwidth = self.amp_bandwidth_factor / self.length_array
        self.bandwidth_overlap_sled = [min(x / self.bandwidth_overlap_sled_factor,
                                           1) for x in self.amp_bandwidth]
        self.power_overlap = self.power_overlap_factor * \
                             np.array(self.bandwidth_overlap_sled)

    def calculate_pump_power_per_mode(self, length, loss) -> float:
        """Calculate the pump power per mode at length

        Returns
        -------
        pump_power_per_mode: float
            pump power per mode after specific length
        """
        initial_pump_power = self.initial_pump_power / self.number_of_modes
        if length == self.length_min:
            pump_power_per_mode = initial_pump_power
        else:
            pump_power_per_mode = 10 ** (
                    -(length - self.length_min) * self.losses_pump /
                    100) * initial_pump_power - loss
        return pump_power_per_mode

    def calculate_gain_per_mode(self, pump_power_per_mode, length, index):
        if length == self.length_min:
            gain_per_mode = np.sqrt(pump_power_per_mode) * self.gain_inside * \
                            self.length_min
        else:
            gain_per_mode = np.sqrt(pump_power_per_mode) * self.gain_inside * (
                    length - self.length_array[index - 1])
        return gain_per_mode

    def calculate_losses_in_amp_band(self, gain_per_mode, length, old) -> float:
        """Calculate the losses & and amplification in the amplification band

        Returns
        -------
        losses_band: float
            losses & amplification in the amplification band
        """
        if length == self.length_min:
            losses_band = np.sinh(gain_per_mode) ** 2 + 1
        else:
            losses_band = (np.sinh(gain_per_mode + old) ** 2 + 1) * 10 ** (-(
                    length - self.length_min) * self.losses_sled / 100)
        return losses_band

    def calculate_power_after_crystal(self, losses_band, length, index) -> float:
        """Calculate the power after the crystal

        Returns
        -------
        power_after_crystal: float
            power after crystal
        """
        if length == self.length_min:
            power_after_crystal = losses_band * self.power_overlap_factor
        else:
            power_after_crystal = losses_band * self.power_overlap[index] + (
                    self.power_overlap_factor - self.power_overlap[index]) * 10 ** (
                                          -(length - self.length_min) *
                                          self.losses_sled / 100)

        return power_after_crystal

    def get_power_after_crystal_list(self) -> list:
        power_result = []
        old_gain = 0
        loss = 0
        pump = []
        for index, length in enumerate(self.length_array):
            pump_power_per_mode = self.calculate_pump_power_per_mode(
                length=length,
                loss=loss)
            print(pump_power_per_mode)
            gain_per_mode = self.calculate_gain_per_mode(
                pump_power_per_mode=pump_power_per_mode, length=length, index=index)
            losses_band = self.calculate_losses_in_amp_band(
                gain_per_mode=gain_per_mode,
                length=length, old=old_gain)
            old_gain += gain_per_mode
            power_after_crystal = self.calculate_power_after_crystal(
                losses_band=losses_band,
                length=length,
                index=index)
            power_result.append(power_after_crystal)
            if length == self.length_min:
                loss = 0.1
            elif index == 1:
                loss = power_after_crystal - power_result[-1]
            else:
                loss = power_after_crystal - power_result[-2]
                print(loss)
        return power_result

    def plot_amp_loss_surface(self, modes: list = [1, 3],
                              powers: list = np.linspace(0.1, 200, num=50),
                              y_lim: list = [-1, 30], x_lim: list = [0, 50]):
        """Plotting the power after the crystal over input power and crystal length
        as surface plot for different modes

        Parameters
        ----------
        modes: list
            contains the specific numbers of modes to analyse
        powers: list
            contains the different powers to analyse at
        y_lim: list
            contains minimum and maximum y value for plotting
        x_lim: list
            contains minimum and maximum x value for plotting

        Returns
        -------

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(self.length_array, np.linspace(start=0.1,
                                                          stop=200, num=50))
        for mode in modes:
            z = []
            self.number_of_modes = mode
            for power in powers:
                self.initial_pump_power = power
                z.append(self.get_power_after_crystal_list())

            c = ax.plot_surface(x, y, np.array(z), label=f'#Modes: {mode}')
            c._facecolors2d = c._facecolor3d
            c._edgecolors2d = c._edgecolor3d
        plt.legend()
        plt.xlabel('Length [mm]')
        plt.ylabel('Input power [mW]')
        ax.set_zlabel('Power [mW]')
        fig.savefig('amp_loss_surface.pdf')
        plt.show()

    def plot_amp_loss(self, modes: list = [1, 3],
                      powers: list = [0.1, 50, 100],
                      y_lim: list = [-1, 30], x_lim: list = [0, 50]):
        """Plot power after crystal over crystal length for different number of modes
         as line plots

        Parameters
        ----------
        modes: list
            contains the specific numbers of modes to analyse
        powers: list
            contains the different powers to analyse at
        y_lim: list
            contains minimum and maximum y value for plotting
        x_lim: list
            contains minimum and maximum x value for plotting

        Returns
        -------

        """
        fig = plt.figure()
        for mode in modes:
            self.number_of_modes = mode
            for power in powers:
                self.initial_pump_power = power
                plt.plot(self.length_array, self.get_power_after_crystal_list(),
                         label=f'{power} mW, #Modes: {mode}')
        plt.legend()
        plt.ylim(y_lim[0], y_lim[1])
        plt.xlim(x_lim[0], x_lim[1])
        plt.xlabel('Length [mm]')
        plt.ylabel('Power [mW]')
        fig.savefig('amp_loss.pdf')
        plt.show()
