from gain_estimation import GainEstimation

# Initialize the class object here, use default values or explicitly define the parameter
simulation = GainEstimation(losses_pump=1, losses_sled=1, stepsize=0.01, length_max=50)

# Plot power after crystal plot
simulation.plot_amplification(modes=[1, 3], powers=[10, 50, 100, 200])

# Plot loss and amplification
simulation.plot_amplification_and_loss(modes=[1, 3], powers=[100, 200])
