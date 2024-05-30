import numpy as np
import matplotlib.pyplot as plt

def adjust_velocities(prev_velocities, inertia_weight, learn_factor1, learn_factor2, max_vel, time_step, p_best_positions, current_positions, global_best_pos):
    random_factor1 = np.random.rand()
    random_factor2 = np.random.rand()

    velocity_increment = (
        inertia_weight * prev_velocities +
        learn_factor1 * random_factor1 * ((p_best_positions - current_positions) / time_step) +
        learn_factor2 * random_factor2 * ((global_best_pos - current_positions) / time_step)
    )

    updated_velocities = np.minimum(velocity_increment, max_vel)
    return updated_velocities

def plot_contour(func, range_vals, const_a):
    x_vals = np.linspace(range_vals[0], range_vals[1], 100)
    y_vals = x_vals

    grid_x, grid_y = np.meshgrid(x_vals, y_vals)
    func_vals = func(grid_x, grid_y)
    log_vals = np.log(const_a + func_vals)
    
    plt.contour(grid_x, grid_y, log_vals, 100)
    plt.show()

def set_initial_positions(num_particles, num_vars, min_limit, max_limit):
    pos_array = min_limit + np.random.rand(num_particles, num_vars) * (max_limit - min_limit)
    return pos_array

def set_initial_velocities(num_particles, num_vars, min_limit, max_limit, coeff, time_step):
    velocity_span = coeff / time_step * (-(max_limit - min_limit) / 2 + np.random.rand(num_particles, num_vars) * (max_limit - min_limit))
    return velocity_span

def objective_function(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def run_pso():
    num_particles = 30
    num_variables = 2
    lower_bound = -5
    upper_bound = 5
    coefficient = 1
    time_increment = 1
    learning_factor1 = 2
    learning_factor2 = 2
    inertia_weight = 1.4
    min_inertia_weight = 0.4
    inertia_weight_reduction = 0.98
    max_velocity = 0.5
    iterations = 10000

    global_best_locations = np.zeros((4, num_variables))
    plot_contour(objective_function, [-5, 5], 0.01)
    plt.ion()
    plt.show()

    best_point_count = 0
    iteration_counter = 1

    while best_point_count < 4:
        positions = set_initial_positions(num_particles, num_variables, lower_bound, upper_bound)
        velocities = set_initial_velocities(num_particles, num_variables, lower_bound, upper_bound, coefficient, time_increment)

        global_best_position = positions[0, :]
        personal_best_positions = positions.copy()

        for k in range(iterations):
            personal_best_values = objective_function(personal_best_positions[:, 0], personal_best_positions[:, 1])
            current_values = objective_function(positions[:, 0], positions[:, 1])
            improved_indices = current_values < personal_best_values
            personal_best_positions[improved_indices] = positions[improved_indices]

            minimum_value = np.min(current_values)
            minimum_index = np.argmin(current_values)
            if minimum_value < objective_function(global_best_position[0], global_best_position[1]):
                global_best_position = positions[minimum_index, :]

            velocities = adjust_velocities(velocities, inertia_weight, learning_factor1, learning_factor2, max_velocity, time_increment, personal_best_positions, positions, global_best_position)
            positions = positions + velocities * time_increment

            if inertia_weight > min_inertia_weight:
                inertia_weight *= inertia_weight_reduction

        is_new_best = True
        for pos in global_best_locations:
            if np.allclose(pos, global_best_position, atol=1e-3):
                is_new_best = False
                break

        if is_new_best:
            best_point_count += 1
            global_best_locations[best_point_count - 1, :] = global_best_position
            print(f"Iteration {iteration_counter}, found minima: x = {global_best_position[0]:.5f}, y = {global_best_position[1]:.5f}, f(x,y) = {objective_function(global_best_position[0], global_best_position[1]):.5f}")

            plt.scatter(global_best_position[0], global_best_position[1], c='k')
            plt.draw()
            plt.pause(0.01)

        iteration_counter += 1

    plt.ioff()
    plt.show()
    print("Global Best Locations:")
    print(global_best_locations)

if __name__ == "__main__":
    run_pso()
