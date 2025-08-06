import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set matplotlib to use Arial font to avoid Type 3 font issues
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['pdf.fonttype'] = 42  # Force TrueType fonts to avoid Type 3
plt.rcParams['ps.fonttype'] = 42  # Same for PostScript output


def load_test_results(results_dir):
    """
    Load test result files from the specified directory
    """
    # Find all result files
    result_files = glob.glob(f"{results_dir}/rf_test_*.json")

    if not result_files:
        print(f"No test result files found in {results_dir} directory")
        return []

    # Load all test results
    all_results = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                # Add filename for reference
                result['file_name'] = os.path.basename(file_path)
                all_results.append(result)
                print(f"Loaded {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(all_results)} test result files")
    return all_results


def extract_trajectory_data(results):
    """
    Extract trajectory data for plotting
    """
    trajectory_data = []

    for result in results:
        if 'position_data' not in result:
            continue

        positions = result['position_data']
        file_name = result.get('file_name', '')

        # Extract position data
        for idx, pos in enumerate(positions):
            if not isinstance(pos, dict):
                print(f"Warning: Invalid position data at index {idx} in {file_name}")
                continue

            if 'target' not in pos or not isinstance(pos['target'], dict):
                print(f"Warning: Missing or invalid target data at index {idx} in {file_name}")
                continue

            entry = {
                'file_name': file_name,
                'x': float(pos.get('x', 0)),
                'y': float(pos.get('y', 0)),
                'z': float(pos.get('z', 0)),
                'target_x': float(pos['target'].get('x', 0)),
                'target_y': float(pos['target'].get('y', 0)),
                'target_z': float(pos['target'].get('z', 0)),
                'phase': pos.get('phase', 'unknown'),
                'time': float(pos.get('time', 0)),
                'sequence_index': int(pos.get('sequence_index', 0)),
                'position_index': int(pos.get('position_index', 0))
            }
            trajectory_data.append(entry)

    return pd.DataFrame(trajectory_data)


def create_square_trajectory_plots(trajectory_df, folder_path, results):
    """
    Create 2D plots for square trajectory flights, with each flight in a separate subplot.
    Each data point is represented as a scatter point.
    """
    if trajectory_df.empty:
        print("Warning: No trajectory data available")
        return None

    try:
        # Ensure matplotlib settings are correct for Arial font
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans',
                                           'sans-serif']
        plt.rcParams['pdf.fonttype'] = 42  # Force TrueType fonts
        plt.rcParams['ps.fonttype'] = 42

        # Get unique test file names and sort them
        test_files = sorted(trajectory_df['file_name'].unique())
        num_files = len(test_files)

        if num_files == 0:
            print("No flight data found")
            return None

        print(f"Creating 2D plots for {num_files} square trajectory flights")

        # Create figure with subplots
        fig, axes = plt.subplots(1, num_files, figsize=(5 * num_files, 5))
        fig.suptitle('Square Trajectory Flights - 2D Top View', fontsize=16, fontfamily='Arial')

        # If only one flight, make axes into array for consistent indexing
        if num_files == 1:
            axes = np.array([axes])

        # Extract network conditions
        network_conditions = {}
        for result in results:
            file_name = result.get('file_name')
            rf_conditions = result.get('rf_conditions', {})
            bandwidth = rf_conditions.get('bandwidth_kbps', 0)
            latency = rf_conditions.get('latency_ms', 0)
            packet_loss = rf_conditions.get('packet_loss_rate', 0)

            # Handle None values
            bandwidth = "Baseline" if bandwidth is None else bandwidth
            latency = 0 if latency is None else latency
            packet_loss = 0 if packet_loss is None else packet_loss

            network_conditions[file_name] = {
                'bandwidth': bandwidth,
                'latency': latency,
                'packet_loss': packet_loss
            }

        # Plot each flight
        for i, file_name in enumerate(test_files):
            # Get data for this flight
            flight_data = trajectory_df[trajectory_df['file_name'] == file_name]

            # Sort data by position index
            flight_data = flight_data.sort_values(by=['position_index'])

            # Get network conditions
            net_cond = network_conditions.get(file_name, {'bandwidth': 0, 'latency': 0, 'packet_loss': 0})

            # Set subplot title
            title = f'Flight {i + 1}\n'
            if net_cond['bandwidth'] == "Baseline":
                title += f'Data Rate: {net_cond["bandwidth"]}, '
            else:
                title += f'Data Rate: {net_cond["bandwidth"]}kbps, '
            title += f'Lat: {net_cond["latency"]}ms, Loss: {net_cond["packet_loss"]}%'

            axes[i].set_title(title, fontsize=10, fontfamily='Arial')

            # Plot all data points as scatter points
            scatter = axes[i].scatter(
                flight_data['x'],
                flight_data['y'],
                c=flight_data['sequence_index'],  # Color by sequence index
                cmap='viridis',
                s=20,  # Point size
                alpha=0.8
            )

            # Plot target waypoints
            target_points = flight_data[['target_x', 'target_y', 'sequence_index']].drop_duplicates()
            axes[i].scatter(
                target_points['target_x'],
                target_points['target_y'],
                color='red',
                marker='D',
                s=80,
                alpha=0.8
            )

            # Label waypoints with sequence numbers
            for _, row in target_points.iterrows():
                axes[i].text(row['target_x'], row['target_y'], str(row['sequence_index'] + 1),
                             fontsize=9, ha='center', va='center', color='white', fontfamily='Arial')

            # Draw ideal path between waypoints
            axes[i].plot(
                target_points['target_x'],
                target_points['target_y'],
                'r--',  # Red dashed line
                alpha=0.6
            )

            # Set axis labels and grid
            axes[i].set_xlabel('X Position (m)', fontfamily='Arial')
            axes[i].set_ylabel('Y Position (m)', fontfamily='Arial')
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].set_aspect('equal')  # Equal aspect ratio for top view

            # Ensure tick labels also use Arial
            for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
                label.set_fontfamily('Arial')

        # Add colorbar
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Sequence Index', fontfamily='Arial')

        # Ensure colorbar tick labels use Arial
        for label in cbar.ax.get_xticklabels():
            label.set_fontfamily('Arial')

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.15)

        # Save as PDF with explicit backend specification
        pdf_file = os.path.join(folder_path, "square_trajectory_plots.pdf")
        plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight', backend='pdf')
        print(f"Square trajectory plots saved to: {pdf_file}")

        # Close figure
        plt.close(fig)

        return True

    except Exception as e:
        print(f"Error creating square trajectory plots: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main function - load data and generate square trajectory plots
    """
    # Get input folder path
    folder_path = input("Please enter the folder path containing test result JSON files: ")

    # 1. Load test results
    results = load_test_results(folder_path)

    if not results:
        print("No test result files found. Exiting.")
        return

    # 2. Extract trajectory data
    trajectory_df = extract_trajectory_data(results)
    print(f"Extracted data for {len(trajectory_df['file_name'].unique())} flights")

    # 3. Create square trajectory plots
    create_square_trajectory_plots(trajectory_df, folder_path, results)

    print("\nTrajectory plot generation complete!")
    print(f"PDF file has been saved to: {os.path.join(folder_path, 'square_trajectory_plots.pdf')}")


if __name__ == "__main__":
    main()