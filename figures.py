import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_boid_counts(boid_counts, num_classes, config_name):
    class_names = ['Prey', 'Predator']

    if not os.path.exists("figures"):
        os.makedirs("figures")

    for class_idx in range(num_classes):
        class_counts = [counts[class_idx] for counts in boid_counts]
        plt.plot(class_counts, label=f'{class_names[class_idx]}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Number of Boids')
    plt.title('Number of Boids per Class Over Time')
    plt.legend()
    plt.savefig(f'figures/boid_counts_{config_name}.png')
    plt.show()
    plt.close()

def plot_family_tree(param_dict, family_tree, config_name):
    parameters = ['Separation', 'Alignment', 'Cohesion']

    fig, axes = plt.subplots(len(parameters), 2, figsize=(12, 12))
    
    if len(parameters) == 1:
        axes = np.array([axes])

    prey = [[], [], []]
    predator = [[], [], []]
    
    # Collect the parameters for all boids
    for boid_id, (params, boid_class) in param_dict.items():
        for param_idx in range(params.shape[0]):
            if boid_class == 0:
                prey[param_idx].append(params[param_idx].tolist())
            elif boid_class == 1:
                predator[param_idx].append(params[param_idx].tolist())

    prey = [np.array(prey[i]) for i in range(len(prey))]
    predator = [np.array(predator[i]) for i in range(len(predator))]

    for param_idx in range(len(parameters)):
        if len(prey[param_idx]) > 0:
            # Use kdeplot for density estimation
            sns.kdeplot(x=prey[param_idx][:, 0], y=prey[param_idx][:, 1], fill=True, cmap='Blues', alpha=1, ax=axes[param_idx, 0])
        if len(predator[param_idx]) > 0:
            sns.kdeplot(x=predator[param_idx][:, 0], y=predator[param_idx][:, 1], fill=True, cmap='Reds', alpha=1, ax=axes[param_idx, 1])

        # Draw general movement vectors
        for parent_id, child_id in family_tree:
            parent_params, parent_class = param_dict[parent_id]
            child_params, child_class = param_dict[child_id]

            assert parent_class == child_class

            dx = child_params[param_idx, 0] - parent_params[param_idx, 0]
            dy = child_params[param_idx, 1] - parent_params[param_idx, 1]
            
            if parent_class == 0:
                axes[param_idx, 0].quiver(parent_params[param_idx, 0], parent_params[param_idx, 1], 
                                          dx, dy, angles='xy', scale_units='xy', scale=1, color='k', alpha=0.1)
            elif parent_class == 1:
                axes[param_idx, 1].quiver(parent_params[param_idx, 0], parent_params[param_idx, 1], 
                                          dx, dy, angles='xy', scale_units='xy', scale=1, color='k', alpha=0.1)

        axes[param_idx, 0].set_xlabel(f'{parameters[param_idx]} self')
        axes[param_idx, 0].set_ylabel(f'{parameters[param_idx]} other')
        axes[param_idx, 1].set_xlabel(f'{parameters[param_idx]} self')
        axes[param_idx, 1].set_ylabel(f'{parameters[param_idx]} other')

    plt.tight_layout()
    plt.savefig(f'figures/family_tree_{config_name}.png')
    plt.show()
    plt.close()

def plot_distribution_over_time(quantiles_prey, quantiles_predator, config_name):
    """
    Plot the distribution of parameters for 'Prey' and 'Predator' over time using quantiles.

    Parameters:
    quantiles_prey (dict): Dictionary containing quantiles for 'Prey' parameters.
    quantiles_predator (dict): Dictionary containing quantiles for 'Predator' parameters.

    Returns:
    None (plots the distribution)
    """
    # Define the parameters and their corresponding subplot positions
    parameters = ['Separation Self', 'Separation Other', 'Alignment Self', 'Alignment Other', 'Cohesion Self', 'Cohesion Other']
    subplot_positions_prey = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    subplot_positions_predator = [(0, 1), (0, 0), (1, 1), (1, 0), (2, 1), (2, 0)]

    # Create a figure with subplots
    fig, axs = plt.subplots(6, 2, figsize=(14, 18))

    for i, (param, pos_pray, pos_predator) in enumerate(zip(parameters, subplot_positions_prey, subplot_positions_predator)):
        # Plot 'Prey' quantiles for current parameter
        ax = axs[i, 0]
        
        for q_low, q_high in zip(['q1', 'q5', 'q25'], ['q99', 'q95', 'q75']):
            array_low = np.array(quantiles_prey[q_low])[:, pos_pray[0], pos_pray[1]]
            array_high = np.array(quantiles_prey[q_high])[:, pos_pray[0], pos_pray[1]]
            ax.fill_between(x=range(len(array_low)), y1=array_low, y2=array_high,
                            alpha=0.3, label='', color="tab:blue")
        
        array_median = np.array(quantiles_prey['q50'])[:, pos_pray[0], pos_pray[1]]
        ax.plot(range(len(array_median)), array_median, '-', color="tab:blue", label='Prey Median')

        # Add title, labels, and legend
        ax.set_title(f"Prey - {param}")
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("Parameter Value")

        # Plot 'Predator' quantiles for current parameter
        ax = axs[i, 1]
        
        for q_low, q_high in zip(['q1', 'q5', 'q25'], ['q99', 'q95', 'q75']):
            array_low = np.array(quantiles_predator[q_low])[:, pos_predator[0], pos_predator[1]]
            array_high = np.array(quantiles_predator[q_high])[:, pos_predator[0], pos_predator[1]]
            ax.fill_between(x=range(len(array_low)), y1=array_low, y2=array_high,
                            alpha=0.3, label='', color="tab:orange")
        
        array_median = np.array(quantiles_predator['q50'])[:, pos_predator[0], pos_predator[1]]
        ax.plot(range(len(array_median)), array_median, '-', color="tab:orange", label='Predator Median')

        # Add title, labels, and legend
        ax.set_title(f"Predator - {param}")
        ax.set_xlabel("Simulation step")
        ax.set_ylabel("Parameter Value")

    # Adjust layout and show plot
    fig.suptitle("Distribution of Parameters (Prey and Predator) Over Time", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplot layout
    plt.savefig(f'figures/distribution_parameters_{config_name}.png')
    plt.show()
    plt.close()