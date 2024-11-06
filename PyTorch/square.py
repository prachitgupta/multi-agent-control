import numpy as np

def square_data(num_agents, dist_min_thres):
    # Side length scales with number of agents
    side_length = np.sqrt(max(1.0, num_agents / 4.0))

    perimeter_length = 4 * side_length  # Total perimeter length

    # Generate parameter t to position agents uniformly along the perimeter
    t = np.linspace(0, perimeter_length, num_agents, endpoint=False)

    # Function to compute positions on the square's perimeter based on t
    def perimeter_position(t_values):
        x = np.zeros_like(t_values)
        y = np.zeros_like(t_values)

        side = side_length

        # Bottom edge (from (0, 0) to (side, 0))
        mask = (t_values >= 0) & (t_values < side)
        x[mask] = t_values[mask]
        y[mask] = 0

        # Right edge (from (side, 0) to (side, side))
        mask = (t_values >= side) & (t_values < 2 * side)
        x[mask] = side
        y[mask] = t_values[mask] - side

        # Top edge (from (side, side) to (0, side))
        mask = (t_values >= 2 * side) & (t_values < 3 * side)
        x[mask] = 3 * side - t_values[mask]
        y[mask] = side

        # Left edge (from (0, side) to (0, 0))
        mask = (t_values >= 3 * side) & (t_values < 4 * side)
        x[mask] = 0
        y[mask] = 4 * side - t_values[mask]

        return np.vstack((x, y)).T

    # Compute starting positions
    s_positions = perimeter_position(t)

    # Compute goal positions opposite to starting positions
    t_opposite = (t + 2 * side_length) % perimeter_length
    g_positions = perimeter_position(t_opposite)

    # Initial velocities are zero
    states = np.concatenate([s_positions, np.zeros((num_agents, 2))], axis=1)
    goals = g_positions

    return states, goals