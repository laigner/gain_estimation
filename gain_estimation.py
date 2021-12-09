import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GainEstimation:
    def __init__(
            self,
            losses_pump: float = 1,
            losses_sled: float = 1,
            mode_size_z: float = 0.7,
            mode_size_y: float = 0.7,
            stepsize: float = 1.0,
            length_max: float = 500,
            number_of_modes: int = 1,
            initial_pump_power: float = 0.0,
            power_overlap_factor: float = 1,
            amp_bandwidth_factor: float = 3.2,
            bandwidth_overlap_sled_factor: float = 5.7,
    ):
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
        stepsize: float
            step length in mm
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
        self.pump_intensity = self.initial_pump_power / (
                self.mode_size_z * self.mode_size_y
        )
        self.intensity_factor = 2.6 * 3.6 / (self.mode_size_z * self.mode_size_y)
        self.gain_inside = 0.00336 * np.sqrt(self.intensity_factor)
        self.stepsize = stepsize
        self.length_max = length_max
        self.number_of_modes = number_of_modes
        self.power_overlap_factor = power_overlap_factor
        self.amp_bandwidth_factor = amp_bandwidth_factor
        self.bandwidth_overlap_sled_factor = bandwidth_overlap_sled_factor
        self.length_array = np.linspace(0, length_max, num=int(length_max / stepsize) + 1)
        self.amp_bandwidth = [1 if x == 0 or self.amp_bandwidth_factor / x >= 1 else self.amp_bandwidth_factor / x for x
                              in self.length_array]
        self.bandwidth_overlap_sled = [
            min(x / self.bandwidth_overlap_sled_factor, 1) for x in self.amp_bandwidth
        ]
        self.power_overlap = self.power_overlap_factor * np.array(
            self.bandwidth_overlap_sled
        )

    def calculate_pump_power_per_mode(self, length, loss) -> float:
        """Calculate the pump power per mode at length
        Parameters
        ----------
        length: float
            specific length of crystal

        loss: float
            loss of pump power

        Returns
        -------
        pump_power_per_mode: float
            pump power per mode after specific length
        """
        initial_pump_power = self.initial_pump_power / self.number_of_modes
        if length == 0:
            pump_power_per_mode = initial_pump_power
        else:
            pump_power_per_mode = 10 ** (
                    -length * self.losses_pump / 100
            ) * initial_pump_power - abs(loss)
        return pump_power_per_mode

    def calculate_gain_per_mode(
            self, pump_power_per_mode: float, length: float
    ) -> float:
        """Calculate gain per mode at specific length

        Parameters
        ----------
        pump_power_per_mode: float
            pump power per mode at specific length
        length: float
            specific length of crystal


        Returns
        -------
        gain_per_mode: float
            gain per mode at specific length
        """
        if length == 0:
            gain_per_mode = 0
        else:
            gain_per_mode = (
                    np.sqrt(pump_power_per_mode)
                    * self.gain_inside
                    * self.stepsize
            )
        return gain_per_mode

    def calculate_amplification_in_amp_band(self, gain_per_mode, length, old_gain) -> float:
        """Calculate the losses & and amplification in the amplification band

        Returns
        -------
        losses_band: float
            losses & amplification in the amplification band
        """
        if length == 0:
            losses_band = np.sinh(gain_per_mode) ** 2 + 1
        else:
            losses_band = (np.sinh(gain_per_mode + old_gain) ** 2 + 1) * 10 ** (
                    -length * self.losses_sled / 100
            )
        return losses_band

    def get_amplification_list(self) -> tuple[list[float], list[float]]:
        """returns a list of the power after the crystal and the loss after the crystal

        Returns
        -------

        """
        amplification_result = []
        amplification_and_loss = []
        old_gain = 0
        loss = 0
        for length in self.length_array:
            # calculate pump power per mode
            pump_power_per_mode = self.calculate_pump_power_per_mode(
                length=length, loss=loss
            )
            # avoid negative values
            if pump_power_per_mode < 0:
                pump_power_per_mode = 0
            # calculate gain per mode
            gain_per_mode = self.calculate_gain_per_mode(
                pump_power_per_mode=pump_power_per_mode, length=length
            )
            # sum old gain per mode with new gain per mode
            old_gain += gain_per_mode
            # calculate losses in amp band
            amplification_band = self.calculate_amplification_in_amp_band(gain_per_mode=gain_per_mode, length=length,
                                                                          old_gain=old_gain)

            # define initial loss
            if length == 0:
                loss = 0
                amplification_and_loss.append(amplification_band)
            # calculate loss (difference of power after crystal values)
            else:
                loss = (amplification_band - amplification_result[0])
                if loss > self.initial_pump_power:
                    loss = self.initial_pump_power
                amplification_and_loss.append(amplification_band * 10 ** (-length * self.losses_sled / 100))
            amplification_result.append(amplification_band)

        return amplification_result, amplification_and_loss

    def plot_amplification(self, modes: list = [1], powers: list = [100]):
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
        # create figure
        fig = plt.figure()
        # iterate through input values
        for mode in modes:
            self.number_of_modes = mode
            for power in powers:
                self.initial_pump_power = power
                # plot the results
                plt.plot(
                    self.length_array,
                    self.get_amplification_list()[0],
                    label=f"{power} mW, #Modes: {mode}",
                )
        plt.legend()
        plt.ylim(0, powers[-1])
        plt.xlim(0, self.length_max)
        plt.xlabel("Length [mm]")
        plt.ylabel("Power [mW]")
        # save figure
        fig.savefig("amp_loss.pdf")
        plt.show()

    def plot_amplification_and_loss(self, modes: list = [1], powers: list = [100]):
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
        # create figure
        fig = plt.figure()
        # iterate through input values
        for mode in modes:
            self.number_of_modes = mode
            for power in powers:
                self.initial_pump_power = power
                # plot the results
                plt.plot(
                    self.length_array,
                    self.get_amplification_list()[1],
                    label=f"{power} mW, #Modes: {mode}",
                )
        plt.legend()
        plt.ylim(0, powers[-1])
        plt.xlim(0, self.length_max)
        plt.xlabel("Length [mm]")
        plt.ylabel("Power [mW]")
        # save figure
        fig.savefig("amp_loss.pdf")
        plt.show()

    # def plot_amp_loss_surface(self, modes: list = [1, 3],
    #                           powers: list = np.linspace(0.1, 200, num=50),
    #                           y_lim: list = [-1, 30], x_lim: list = [0, 50]):
    #     """Plotting the power after the crystal over input power and crystal length
    #     as surface plot for different modes
    #
    #     Parameters
    #     ----------
    #     modes: list
    #         contains the specific numbers of modes to analyse
    #     powers: list
    #         contains the different powers to analyse at
    #     y_lim: list
    #         contains minimum and maximum y value for plotting
    #     x_lim: list
    #         contains minimum and maximum x value for plotting
    #
    #     Returns
    #     -------
    #
    #     """
    #     #create figure
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     x, y = np.meshgrid(self.length_array, np.linspace(start=0.1,
    #                                                       stop=200, num=50))
    #     for mode in modes:
    #         z = []
    #         self.number_of_modes = mode
    #         for power in powers:
    #             self.initial_pump_power = power
    #             z.append(self.get_power_after_crystal_list())
    #
    #         c = ax.plot_surface(x, y, np.array(z), label=f'#Modes: {mode}')
    #         c._facecolors2d = c._facecolor3d
    #         c._edgecolors2d = c._edgecolor3d
    #     plt.legend()
    #     plt.xlabel('Length [mm]')
    #     plt.ylabel('Input power [mW]')
    #     ax.set_zlabel('Power [mW]')
    #     fig.savefig('amp_loss_surface.pdf')
    #     plt.show()
