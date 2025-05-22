import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import glob
import os
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Any


def load_test_results(json_file):
    """
    Load test results from a single JSON file
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def load_all_test_results(directory_or_files):
    """
    Load all test results from a directory or a list of files

    Parameters:
    - directory_or_files: Directory path or list of file paths

    Returns:
    - List of loaded test result data
    """
    results = []

    # Check if input is a directory
    if isinstance(directory_or_files, str) and os.path.isdir(directory_or_files):
        # Find all JSON files in the directory
        json_files = glob.glob(os.path.join(directory_or_files, "*.json"))
        print(f"Found {len(json_files)} JSON files in {directory_or_files}")
    elif isinstance(directory_or_files, list):
        # Use provided list of files
        json_files = directory_or_files
        print(f"Processing {len(json_files)} provided JSON files")
    else:
        # Single file
        json_files = [directory_or_files]
        print(f"Processing single file: {directory_or_files}")

    # Load each file
    for file_path in json_files:
        try:
            result = load_test_results(file_path)
            if result:
                # Add filename to result for reference
                result['file_name'] = os.path.basename(file_path)

                # Verify this is a square trajectory test
                sequence = result.get('sequence', [])
                if sequence and len(sequence) >= 4:
                    # Check if it forms a rough square by looking at XY coordinates
                    x_coords = [p[0] for p in sequence]
                    y_coords = [p[1] for p in sequence]

                    if len(set(x_coords)) > 1 and len(set(y_coords)) > 1:
                        results.append(result)
                        print(f"Successfully loaded square trajectory: {file_path}")
                    else:
                        print(f"Skipped non-square trajectory: {file_path}")
                else:
                    print(f"Skipped file with insufficient waypoints: {file_path}")
            else:
                print(f"Failed to load: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Successfully loaded {len(results)} square trajectory test files")
    return results


def extract_trajectory_data(results_list):
    """
    Extract trajectory-related data from multiple test results for square trajectory

    Parameters:
    - results_list: List of test result data

    Returns:
    - Dictionary with combined trajectory data
    """
    # Initialize containers for all data points
    combined_data = {
        'overall_errors': [],
        'xy_errors': [],
        'height_errors': [],
        'transit_errors': [],
        'waypoint_errors': [],
        'command_stats': {
            'sent': 0,
            'dropped': 0,
            'total_attempts': 0
        },
        'rf_conditions': {
            'bandwidth_kbps': None,
            'latency_ms': None,
            'packet_loss_rate': None
        }
    }

    # Extract RF conditions from first file (assuming all files have same conditions)
    if results_list:
        first_result = results_list[0]
        rf_cond = first_result.get('rf_conditions', {})
        combined_data['rf_conditions']['bandwidth_kbps'] = rf_cond.get('bandwidth_kbps')
        combined_data['rf_conditions']['latency_ms'] = rf_cond.get('latency_ms')
        combined_data['rf_conditions']['packet_loss_rate'] = rf_cond.get('packet_loss_rate')

    # Process each result file
    for result in results_list:
        # Extract position data
        position_data = result.get('position_data', [])

        # Get all points
        for point in position_data:
            combined_data['overall_errors'].append(point.get('error', 0))
            combined_data['xy_errors'].append(point.get('error_xy', 0))
            combined_data['height_errors'].append(point.get('error_z', 0))

            # Categorize by phase
            phase = point.get('phase', '')
            if phase == 'transit':
                combined_data['transit_errors'].append(point.get('error', 0))
            elif phase == 'waypoint':
                combined_data['waypoint_errors'].append(point.get('error', 0))

        # Aggregate command stats
        command_stats = result.get('command_stats', {})
        combined_data['command_stats']['sent'] += command_stats.get('sent', 0)
        combined_data['command_stats']['dropped'] += command_stats.get('dropped', 0)
        combined_data['command_stats']['total_attempts'] += command_stats.get('total_attempts', 0)

    return combined_data


def create_square_trajectory_analysis(results_list, output_file=None):
    """
    Create an error analysis chart for square trajectory tests

    Parameters:
    - results_list: List of test result data
    - output_file: Output file path for saving the chart
    """
    if not results_list:
        print("No results to analyze")
        return

    # Make sure figure size is set properly from the beginning
    plt.rcParams["figure.figsize"] = (20, 6)
    plt.rcParams["figure.autolayout"] = False

    # Extract trajectory data from all results
    combined_data = extract_trajectory_data(results_list)

    # Get error data and command stats
    overall_errors = combined_data['overall_errors']
    xy_errors = combined_data['xy_errors']
    height_errors = combined_data['height_errors']
    transit_errors = combined_data['transit_errors']
    waypoint_errors = combined_data['waypoint_errors']

    command_stats = combined_data['command_stats']
    sent = command_stats.get('sent', 0)
    dropped = command_stats.get('dropped', 0)
    total_attempts = command_stats.get('total_attempts', 0)

    # Fix the success rate calculation
    if total_attempts > 0:
        # Using total_attempts as reference for success rate is more accurate
        success_rate = (sent / total_attempts) * 100 if total_attempts > 0 else 0
    else:
        # Fallback calculation
        success_rate = ((sent - dropped) / sent) * 100 if sent > 0 else 0

    # Get RF conditions
    rf_cond = combined_data['rf_conditions']
    bandwidth = rf_cond.get('bandwidth_kbps', 0)
    if not bandwidth:
        bandwidth = "Baseline"
    latency = rf_cond.get('latency_ms', 0)
    if not latency:
        latency = 0
    packet_loss = rf_cond.get('packet_loss_rate', 0)
    if not packet_loss:
        packet_loss = 0

    # Create a figure for the combined error analysis
    fig = plt.figure(figsize=(20, 6))
    if bandwidth == "Baseline":
        title = f'Square Trajectory Analysis (BW: {bandwidth}, Latency: {latency}ms, Loss: {packet_loss}%)'
    else:
        title = f'Square Trajectory Analysis (BW: {bandwidth}kbps, Latency: {latency}ms, Loss: {packet_loss}%)'
    fig.suptitle(title, fontsize=14)

    # Set up grid for plots in a single row
    gs = GridSpec(1, 6, figure=fig)

    # 1. Error boxplot
    ax1 = fig.add_subplot(gs[0, 0])
    boxplot_data = [overall_errors, xy_errors, height_errors]
    labels = ['Overall', 'XY', 'Z']
    box = ax1.boxplot(boxplot_data, tick_labels=labels, patch_artist=True, widths=0.7)

    # Beautify boxplot
    colors = ['lightblue', 'lightgreen', 'lightpink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Make whiskers and median lines more visible
    for whisker in box['whiskers']:
        whisker.set_linewidth(1.5)
        whisker.set_color('black')

    for median in box['medians']:
        median.set_linewidth(2)
        median.set_color('darkred')

    ax1.set_ylabel('Error (m)', fontsize=9)
    ax1.set_title('Error Distribution', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=8)

    # 2. Error bar chart
    ax2 = fig.add_subplot(gs[0, 1])

    # Calculate means and standard deviations
    mean_overall = np.mean(overall_errors) if overall_errors else 0
    std_overall = np.std(overall_errors) if overall_errors else 0

    mean_xy = np.mean(xy_errors) if xy_errors else 0
    std_xy = np.std(xy_errors) if xy_errors else 0

    mean_z = np.mean(height_errors) if height_errors else 0
    std_z = np.std(height_errors) if height_errors else 0

    means = [mean_overall, mean_xy, mean_z]
    stds = [std_overall, std_xy, std_z]

    x = np.arange(len(labels))
    ax2.bar(x, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Error (m)', fontsize=9)
    ax2.set_title('Avg Error with StdDev', fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=8)

    # Add mean value labels
    for i, v in enumerate(means):
        ax2.text(i, v + stds[i] + 0.001, f"{v:.4f}m", ha='center',
                 fontsize=9, fontweight='bold')

    ax2.grid(True, linestyle='--', alpha=0.7)

    # 3. Path vs Waypoint Errors comparison
    ax3 = fig.add_subplot(gs[0, 2])

    # Prepare data for bar chart
    mean_transit = np.mean(transit_errors) if transit_errors else 0
    std_transit = np.std(transit_errors) if transit_errors else 0

    mean_waypoint = np.mean(waypoint_errors) if waypoint_errors else 0
    std_waypoint = np.std(waypoint_errors) if waypoint_errors else 0

    path_means = [mean_transit, mean_waypoint]
    path_stds = [std_transit, std_waypoint]
    path_labels = ['Path\nFollowing', 'Corner\nPrecision']

    # Create bar chart
    x = np.arange(len(path_labels))
    ax3.bar(x, path_means, yerr=path_stds, align='center', alpha=0.7,
            ecolor='black', capsize=5, color=['skyblue', 'lightgreen'])

    ax3.set_xticks(x)
    ax3.set_xticklabels(path_labels)
    ax3.set_ylabel('Error (m)', fontsize=9)
    ax3.set_title('Path vs Corner Precision', fontsize=10)

    # Add value labels
    for i, v in enumerate(path_means):
        ax3.text(i, v + path_stds[i] + 0.001, f"{v:.4f}m", ha='center',
                 fontsize=9, fontweight='bold')

    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.tick_params(axis='both', which='major', labelsize=8)

    # 4. Command success rate
    ax4 = fig.add_subplot(gs[0, 3])

    if total_attempts > 0:
        ax4.bar(['Success'], [success_rate], color='green', alpha=0.7)
        ax4.set_ylim([0, 105])  # Leave room for 100%
        ax4.set_ylabel('Rate (%)', fontsize=9)
        ax4.set_title('Command Success', fontsize=10)

        # Make percentage more visible
        ax4.text(0, success_rate + 2, f"{success_rate:.2f}%",
                 ha='center', fontsize=10, fontweight='bold')

        # Add simplified command statistics with improved readability
        stats_info = f"{sent}/{total_attempts}"
        ax4.text(0, success_rate / 2 + 15, stats_info,
                 ha='center', fontsize=9, fontweight='bold')

        # Add dropped packets on a new line with more space
        dropped_info = f"Dropped: {dropped}"
        ax4.text(0, success_rate / 2 - 15, dropped_info,
                 ha='center', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax4.transAxes, fontsize=8)

    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.tick_params(axis='both', which='major', labelsize=8)

    # 5. Overall Error Distribution Histogram
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.hist(overall_errors, bins=15, alpha=0.7, color='blue')
    ax5.set_xlabel('Error (m)', fontsize=9)
    ax5.set_ylabel('Count', fontsize=9)
    ax5.set_title('Overall Error Histogram', fontsize=10)
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.tick_params(axis='both', which='major', labelsize=8)

    # Add minimal statistics text
    if overall_errors:
        stats_text = (f"Mean: {mean_overall:.4f}m\n"
                      f"Med: {np.median(overall_errors):.4f}m")
        ax5.text(0.95, 0.95, stats_text, transform=ax5.transAxes,
                 fontsize=8, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))

    # 6. Transit vs Waypoint Error Histogram
    ax6 = fig.add_subplot(gs[0, 5])

    if transit_errors:
        ax6.hist(transit_errors, bins=15, alpha=0.6, color='skyblue', label='Path')
    if waypoint_errors:
        ax6.hist(waypoint_errors, bins=15, alpha=0.6, color='lightgreen', label='Corner')

    ax6.set_xlabel('Error (m)', fontsize=9)
    ax6.set_ylabel('Count', fontsize=9)
    ax6.set_title('Path vs Corner Errors', fontsize=10)
    ax6.legend(fontsize=8, loc='upper right')
    ax6.grid(True, linestyle='--', alpha=0.7)
    ax6.tick_params(axis='both', which='major', labelsize=8)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if output file is specified
    if output_file:
        try:
            fig.savefig(output_file, bbox_inches='tight', dpi=300)
            print(f"Square trajectory analysis chart saved to: {output_file}")
        except Exception as e:
            print(f"Error saving figure to {output_file}: {e}")
            # Try alternate save method
            try:
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                print(f"Square trajectory analysis chart saved using alternate method to: {output_file}")
            except Exception as e2:
                print(f"Failed to save figure using alternate method: {e2}")

    # Display figure
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics (Square Trajectory) ===")
    print(f"Files analyzed: {len(results_list)}")
    print(f"Total data points: {len(overall_errors)}")

    print("\nOverall Error:")
    print(f"  Mean: {mean_overall:.4f} m")
    print(f"  Median: {np.median(overall_errors):.4f} m")
    print(f"  StdDev: {std_overall:.4f} m")

    print("\nXY Plane Error:")
    print(f"  Mean: {mean_xy:.4f} m")
    print(f"  Median: {np.median(xy_errors):.4f} m")
    print(f"  StdDev: {std_xy:.4f} m")

    print("\nHeight (Z) Error:")
    print(f"  Mean: {mean_z:.4f} m")
    print(f"  Median: {np.median(height_errors):.4f} m")
    print(f"  StdDev: {std_z:.4f} m")

    print("\nPath Following (Transit Phase):")
    print(f"  Mean: {mean_transit:.4f} m")
    print(f"  Median: {np.median(transit_errors):.4f} m")
    print(f"  StdDev: {std_transit:.4f} m")

    print("\nCorner Precision (Waypoint Phase):")
    print(f"  Mean: {mean_waypoint:.4f} m")
    print(f"  Median: {np.median(waypoint_errors):.4f} m")
    print(f"  StdDev: {std_waypoint:.4f} m")

    print("\nCommand Success Rate:")
    print(f"  {success_rate:.2f}% ({sent}/{total_attempts})")
    print(f"  Dropped packets: {dropped}")

    return fig


def main():
    """
    Main function to generate square trajectory analysis chart
    """
    # Request user input for the directory or files
    user_input = input("Please enter the path to the directory containing square trajectory test JSON files: ")

    # Check if it's a directory or file
    if os.path.isdir(user_input):
        # It's a directory, load all JSON files
        results = load_all_test_results(user_input)
    elif os.path.isfile(user_input):
        # It's a single file
        results = load_all_test_results(user_input)
    else:
        # Try to interpret as a glob pattern
        try:
            files = glob.glob(user_input)
            if files:
                results = load_all_test_results(files)
            else:
                print(f"No files found matching pattern: {user_input}")
                return
        except Exception as e:
            print(f"Error interpreting input: {e}")
            return

    if not results:
        print("No square trajectory results to analyze")
        return

    # Generate output file path
    if os.path.isdir(user_input):
        output_dir = user_input
    else:
        output_dir = os.path.dirname(user_input)

    # Create the output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {e}")

    output_file = os.path.join(output_dir, "square_trajectory_analysis.pdf")

    # Create trajectory analysis chart
    create_square_trajectory_analysis(results, output_file)


if __name__ == "__main__":
    main()