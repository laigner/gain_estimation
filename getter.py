from gain_estimation import GainEstimation

# Initialize the class object here, use default values or explicitly define the parameter
simulation = GainEstimation()


# Plot power after crystal plot
simulation.plot_amp_loss(modes=[1, 3], powers=[10, 50, 100, 200])

# Plot loss
# simulation.plot_loss_pump(modes=[1,3], powers=[100, 200])
