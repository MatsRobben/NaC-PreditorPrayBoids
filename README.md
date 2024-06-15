# NaC-PreditorPrayBoids

This is the repository for the 2023-2024 Natural Computing course project. You can find the repository at the following link: [NaC-PreditorPrayBoids](https://github.com/MatsRobben/NaC-PreditorPrayBoids.git)

## Installation

To clone the repository and install the necessary requirements, follow these steps:

```bash
# Clone the repository
git clone https://github.com/MatsRobben/NaC-PreditorPrayBoids.git

# Navigate to the project directory
cd NaC-PreditorPrayBoids

# Install the required packages
pip install -r requirements.txt
```

## File Descriptions

`boids.py`

This file contains the main simulation code. Key variables include:

- `EVOLUTION`: 
  - `True` - Runs the simulation with local parameters.
  - `False` - Runs the simulation with global parameters.
- `load_data`:
  - Used to load stored statistical data. If set to `True`, the simulation will not run again and only the figures will be plotted.
- `visual`:
  - Toggles the Pygame visualization. If set to `True`, the simulation visualization will be shown; if `False`, it will not.

The configuration settings in `boids.py` can be found at the top of the file (line 17).

`figures.py`

This file contains all the functions used for plotting the results of the simulation.

`make_config.py`

This file is used to create and store configuration files. This functionality makes it easy to keep track of all the different configurations that are used in the simulation.

## Usage

1. **Running the Simulation**:
   - Adjust the `EVOLUTION`, `load_data`, and `visual` variables in `boids.py` as needed.
   - Ensure your desired configuration is set at the top of `boids.py` (line 17).
   - Run the simulation by executing `boids.py`.

2. **Plotting Figures**:
   - If `load_data` is set to `True` in `boids.py`, the script will only plot the figures using the stored data.

3. **Creating Configuration Files**:
   - Use `make_config.py` to create and store different configuration files as needed for your simulations.

## Contributing

If you would like to contribute to this project, please fork the repository and create a pull request. For major changes, please open an issue first to discuss what you would like to change.
