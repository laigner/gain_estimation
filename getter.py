from gain_estimation import GainEstimation

# Initialize the class object here, use default values or explicitly define the parameter
simulation = GainEstimation(initial_pump_power=200, number_of_modes=1, losses_sled= 0)

# now one can easily access the parameter or methods defined in gain_estimation.py

# e.g.

# Just printing the set initial_pump_power
# print(simulation.initial_pump_power)

# Printing a list containing the values of the power after the crystal
# print(simulation.calculate_power_after_crystal())

# Plot a surface plot of input power output and length
# simulation.plot_amp_loss_surface()

# Plot normal plot
# simulation.plot_amp_loss()

result = simulation.get_power_after_crystal_list()
#print(result)
simulation.plot_amp_loss()
#simulation.plot_amp_loss_surface()