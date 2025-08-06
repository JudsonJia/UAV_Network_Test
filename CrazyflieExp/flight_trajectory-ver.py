import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

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


def create_matplotlib_trajectory_plots(trajectory_df, folder_path, results):
    """
    Create separate trajectory plots using Matplotlib, with each flight in an independent coordinate system
    Save as PDF file, and add network condition and flight mode information
    Files are processed in the order they appear in the file system (not alphabetically sorted)
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

        # Get unique test file names in the order they were loaded (NOT sorted)
        # This preserves the file system order
        test_files = []
        seen_files = set()
        for result in results:
            file_name = result.get('file_name', '')
            if file_name and file_name not in seen_files:
                test_files.append(file_name)
                seen_files.add(file_name)

        num_files = len(test_files)

        if num_files == 0:
            print("No flight data found")
            return None

        print(f"Creating separated trajectory plots for {num_files} flights using Matplotlib")
        print("File processing order:")
        for i, file_name in enumerate(test_files):
            print(f"  {i + 1}. {file_name}")

        # Extract network condition information
        network_conditions = {}
        flight_modes = {}
        for result in results:
            file_name = result.get('file_name', '')

            # Extract network conditions
            rf_conditions = result.get('rf_conditions', {})
            bandwidth = rf_conditions.get('bandwidth_kbps', 0)
            latency = rf_conditions.get('latency_ms', 0)
            packet_loss = rf_conditions.get('packet_loss_rate', 0)

            # Ensure all values have default value 0
            bandwidth = bandwidth if bandwidth is not None else "Baseline"
            latency = latency if latency is not None else 0
            packet_loss = packet_loss if packet_loss is not None else 0

            network_conditions[file_name] = {
                'bandwidth': bandwidth,
                'latency': latency,
                'packet_loss': packet_loss
            }

            # Determine flight mode
            flight_mode = "Unknown"
            if 'sequence' in result:
                sequence = result.get('sequence', [])

                if len(sequence) >= 3:
                    # Check if all points have the same x and y coordinates (vertical ascent/descent)
                    is_vertical = all(point[0] == sequence[0][0] and point[1] == sequence[0][1] for point in sequence)

                    # Check if the path forms a square
                    is_square = False
                    if not is_vertical:
                        # Square trajectory typically has 4 or more points, and the same z coordinate
                        same_height = all(abs(point[2] - sequence[0][2]) < 0.1 for point in sequence)
                        # Simple check: trajectory formed by 4 points
                        if same_height and len(sequence) >= 4:
                            is_square = True

                    if is_vertical:
                        flight_mode = "Vertical"
                    elif is_square:
                        flight_mode = "Square"
                    else:
                        flight_mode = "Other"

            flight_modes[file_name] = flight_mode

        # Find the coordinate range of all trajectory points and target points for unified axes
        all_x = trajectory_df['x'].tolist() + trajectory_df['target_x'].tolist()
        all_y = trajectory_df['y'].tolist() + trajectory_df['target_y'].tolist()
        all_z = trajectory_df['z'].tolist() + trajectory_df['target_z'].tolist()

        # Calculate center point and range
        x_center = np.mean(all_x)
        y_center = np.mean(all_y)
        z_center = np.mean(all_z)

        # Enlarge coordinate axes to make errors less obvious
        # Calculate coordinate range and expand by 1.5 times
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        z_range = max(all_z) - min(all_z)

        # Ensure range is not zero
        x_range = max(x_range, 0.1)
        y_range = max(y_range, 0.1)
        z_range = max(z_range, 0.1)

        # Expand range
        scale_factor = 1.5
        x_min = x_center - x_range * scale_factor / 2
        x_max = x_center + x_range * scale_factor / 2
        y_min = y_center - y_range * scale_factor / 2
        y_max = y_center + y_range * scale_factor / 2
        z_min = z_center - z_range * scale_factor / 2
        z_max = z_center + z_range * scale_factor / 2

        # Create a large figure, increase height to accommodate colorbar and title
        fig = plt.figure(figsize=(5 * num_files, 7))
        fig.suptitle('UAV Flight Trajectories Comparison', fontsize=16, fontfamily='Arial')

        # Create a subplot for each flight in file system order
        for i, file_name in enumerate(test_files):
            # Get data for this flight
            flight_data = trajectory_df[trajectory_df['file_name'] == file_name]

            # Sort data to ensure continuous trajectory
            flight_data = flight_data.sort_values(by=['position_index'])

            # Create subplot
            ax = fig.add_subplot(1, num_files, i + 1, projection='3d')

            # Get network conditions and flight mode
            net_cond = network_conditions.get(file_name, {'bandwidth': 0, 'latency': 0, 'packet_loss': 0})
            mode = flight_modes.get(file_name, "Unknown")

            # Set subplot title, including network conditions and flight mode
            subtitle = f'Flight {i + 1}\n'
            subtitle += f'Mode: {mode}\n'
            if net_cond['bandwidth'] == "Baseline":
                subtitle += f'Data Rate: {net_cond["bandwidth"]}, '
            else:
                subtitle += f'Data Rate: {net_cond["bandwidth"]}kbps, '
            subtitle += f'Latency: {net_cond["latency"]}ms, '
            subtitle += f'Loss: {net_cond["packet_loss"]}%'

            ax.set_title(subtitle, fontsize=10, fontfamily='Arial')

            # Add trajectory line
            ax.plot(flight_data['x'], flight_data['y'], flight_data['z'], 'b-', linewidth=2)

            # Add trajectory points
            scatter = ax.scatter(
                flight_data['x'],
                flight_data['y'],
                flight_data['z'],
                c=flight_data['sequence_index'],
                cmap='viridis',
                s=30,
                alpha=0.8
            )

            # Add target points
            target_points = flight_data[['target_x', 'target_y', 'target_z', 'sequence_index']].drop_duplicates()
            ax.scatter(
                target_points['target_x'],
                target_points['target_y'],
                target_points['target_z'],
                color='red',
                marker='D',
                s=100,
                alpha=1.0
            )

            # Set coordinate axis range
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            # Set coordinate axis labels with Arial font
            ax.set_xlabel('X Position (m)', fontfamily='Arial')
            ax.set_ylabel('Y Position (m)', fontfamily='Arial')
            ax.set_zlabel('Z Position (m)', fontfamily='Arial')

            # Ensure tick labels use Arial font
            for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
                label.set_fontfamily('Arial')

            # Adjust view angle
            ax.view_init(elev=30, azim=45)

        # Manually adjust subplot position to leave space for colorbar
        # Note: tight_layout() is not used because it's incompatible with 3D plots
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95, wspace=0.3)

        # Add color bar, place at the bottom to make it more horizontal and wide
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Sequence Index', fontfamily='Arial')

        # Ensure colorbar tick labels use Arial font
        for label in cbar.ax.get_xticklabels():
            label.set_fontfamily('Arial')

        # Save as PDF with explicit backend specification
        pdf_file = os.path.join(folder_path, "flight_trajectories.pdf")
        plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight', backend='pdf')
        print(f"Trajectory plots saved to PDF: {pdf_file}")

        # Close figure to release memory
        plt.close(fig)

        return True

    except Exception as e:
        print(f"Error creating trajectory plots: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main function - Run the trajectory plot generation process
    """
    # Get input folder path
    folder_path = input("Please enter the folder path containing test result JSON files: ")

    # 1. Load all test results
    results = load_test_results(folder_path)

    if not results:
        print("No test result files found. Exiting.")
        return

    # 2. Extract trajectory data
    trajectory_df = extract_trajectory_data(results)
    print(f"Extracted data for {len(trajectory_df['file_name'].unique())} flights")

    # 3. Create separate trajectory plots and save as PDF
    create_matplotlib_trajectory_plots(trajectory_df, folder_path, results)

    print("\nTrajectory plots generation complete!")
    print("PDF file has been generated in the same folder.")


if __name__ == "__main__":
    main()