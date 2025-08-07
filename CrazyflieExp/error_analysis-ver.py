import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import glob
import os
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Any

# Set matplotlib to use Arial font to avoid Type 3 font issues
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['pdf.fonttype'] = 42  # Force TrueType fonts to avoid Type 3
plt.rcParams['ps.fonttype'] = 42   # Same for PostScript output


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
                results.append(result)
                print(f"Successfully loaded: {file_path}")
            else:
                print(f"Failed to load: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Successfully loaded {len(results)} test result files")
    return results


def extract_stable_phase_data(results_list):
    """
    Extract stable phase data from multiple test results

    Parameters:
    - results_list: List of test result data

    Returns:
    - Dictionary with combined stable phase data
    """
    # Initialize containers for all stable phase data points
    combined_data = {
        'overall_errors': [],
        'xy_errors': [],
        'height_errors': [],
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

        # Filter for stable phase data points
        stable_points = [p for p in position_data if p.get('phase') == 'waypoint' and p.get('stabilized', False)]

        # Extract errors from stable points
        for point in stable_points:
            combined_data['overall_errors'].append(point.get('error', 0))
            combined_data['xy_errors'].append(point.get('error_xy', 0))
            combined_data['height_errors'].append(point.get('error_z', 0))

        command_stats = result.get('command_stats', {})
        combined_data['command_stats']['sent'] += command_stats.get('sent', 0)
        combined_data['command_stats']['dropped'] += command_stats.get('dropped', 0)
        combined_data['command_stats']['total_attempts'] += command_stats.get('total_attempts', 0)

    return combined_data


def generate_output_filename(rf_conditions, base_name="average_error_analysis"):
    """
    Generate output filename based on RF conditions

    Parameters:
    - rf_conditions: Dictionary with RF condition parameters
    - base_name: Base filename

    Returns:
    - Formatted filename with parameters
    """
    filename_parts = [base_name]

    # Check if it's baseline conditions (no RF impairments)
    bandwidth = rf_conditions.get('bandwidth_kbps')
    latency = rf_conditions.get('latency_ms')
    packet_loss = rf_conditions.get('packet_loss_rate')

    # If all conditions are baseline/default, add _baseline
    if (not bandwidth or bandwidth == "Baseline") and (not latency or latency == 0) and (
            not packet_loss or packet_loss == 0):
        filename_parts.append("_baseline")
    else:
        # Add bandwidth (only if not baseline)
        if bandwidth and bandwidth != "Baseline":
            if bandwidth >= 1000:
                # For >= 1000, convert to M and handle decimals
                bw_val = bandwidth / 1000
                if bw_val == int(bw_val):
                    filename_parts.append(f"_{int(bw_val)}M")
                else:
                    # Remove trailing zeros
                    filename_parts.append(f"_{bw_val:g}M")
            else:
                # For < 1000, keep as k and handle decimals
                if bandwidth == int(bandwidth):
                    filename_parts.append(f"_{int(bandwidth)}k")
                else:
                    # Remove trailing zeros
                    filename_parts.append(f"_{bandwidth:g}k")

        # Add latency (only if not 0)
        if latency and latency > 0:
            filename_parts.append(f"_{int(latency)}ms")

        # Add packet loss (only if not 0)
        if packet_loss and packet_loss > 0:
            filename_parts.append(f"_{int(packet_loss)}")

    return ''.join(filename_parts) + '.pdf'


def create_average_error_analysis(results_list, output_file=None):
    """
    Create an error analysis chart based on the average of multiple test results

    Parameters:
    - results_list: List of test result data
    - output_file: Output file path for saving the chart
    """
    if not results_list:
        print("No results to analyze")
        return

    # Ensure matplotlib settings are correct for Arial font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['pdf.fonttype'] = 42  # Force TrueType fonts
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.autolayout"] = False

    # Extract combined stable phase data from all results
    combined_data = extract_stable_phase_data(results_list)

    # Get error data and command stats
    overall_errors = combined_data['overall_errors']
    xy_errors = combined_data['xy_errors']
    height_errors = combined_data['height_errors']

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
    fig = plt.figure(figsize=(10, 6))
    if bandwidth == "Baseline":
        title = f'UAV Performance Analysis (Data Rate: {bandwidth}, Latency: {latency}ms, Loss: {packet_loss}%)'
    else:
        title = f'UAV Performance Analysis (Data Rate: {bandwidth}kbps, Latency: {latency}ms, Loss: {packet_loss}%)'
    fig.suptitle(title, fontsize=14, fontfamily='Arial')

    # Set up grid for plots in a single row - only 3 plots now
    gs = GridSpec(1, 3, figure=fig)

    # 1. Error boxplot (original plot 1)
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

    ax1.set_ylabel('Error (m)', fontsize=9, fontfamily='Arial')
    ax1.set_title('Error Distribution', fontsize=10, fontfamily='Arial')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=8)

    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontfamily('Arial')

    # 2. Command success rate (original plot 3)
    ax2 = fig.add_subplot(gs[0, 1])

    if total_attempts > 0:
        ax2.bar(['Success'], [success_rate], color='green', alpha=0.7)
        ax2.set_ylim([0, 105])  # Leave room for 100%
        ax2.set_ylabel('Rate (%)', fontsize=9, fontfamily='Arial')
        ax2.set_title('Command Success', fontsize=10, fontfamily='Arial')

        # Make percentage more visible
        ax2.text(0, success_rate + 2, f"{success_rate:.2f}%",
                 ha='center', fontsize=10, fontweight='bold', fontfamily='Arial')

        # Add simplified command statistics with improved readability
        stats_info = f"{sent}/{total_attempts}"
        ax2.text(0, success_rate / 2 + 15, stats_info,
                 ha='center', fontsize=9, fontweight='bold', fontfamily='Arial')

        # Add dropped packets on a new line with more space
        dropped_info = f"Dropped: {dropped}"
        ax2.text(0, success_rate / 2 - 15, dropped_info,
                 ha='center', fontsize=9, fontfamily='Arial')
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes,
                fontsize=8, fontfamily='Arial')

    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=8)

    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontfamily('Arial')

    # 3. Overall Error Distribution Histogram (original plot 4)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(overall_errors, bins=15, alpha=0.7, color='blue')
    ax3.set_xlabel('Error (m)', fontsize=9, fontfamily='Arial')
    ax3.set_ylabel('Count', fontsize=9, fontfamily='Arial')
    ax3.set_title('Overall Error Histogram', fontsize=10, fontfamily='Arial')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.tick_params(axis='both', which='major', labelsize=8)

    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_fontfamily('Arial')

    # Add minimal statistics text
    if overall_errors:
        mean_overall = np.mean(overall_errors)
        stats_text = (f"Mean: {mean_overall:.4f}m\n"
                      f"Med: {np.median(overall_errors):.4f}m")
        ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
                 fontsize=8, va='top', ha='right', fontfamily='Arial',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure if output file is specified
    if output_file:
        try:
            # Explicitly specify save parameters to ensure TrueType fonts
            fig.savefig(output_file, bbox_inches='tight', dpi=300, backend='pdf')
            print(f"Combined error analysis chart saved to: {output_file}")
        except Exception as e:
            print(f"Error saving figure to {output_file}: {e}")
            # Try alternate save method
            try:
                plt.savefig(output_file, bbox_inches='tight', dpi=300, backend='pdf')
                print(f"Combined error analysis chart saved using alternate method to: {output_file}")
            except Exception as e2:
                print(f"Failed to save figure using alternate method: {e2}")

    # Display figure
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics (Stable Phase) ===")
    print(f"Files analyzed: {len(results_list)}")
    print(f"Total data points: {len(overall_errors)}")
    print("\nOverall Error:")
    if overall_errors:
        mean_overall = np.mean(overall_errors)
        std_overall = np.std(overall_errors)
        print(f"  Mean: {mean_overall:.4f} m")
        print(f"  Median: {np.median(overall_errors):.4f} m")
        print(f"  StdDev: {std_overall:.4f} m")
    print("\nXY Plane Error:")
    if xy_errors:
        mean_xy = np.mean(xy_errors)
        std_xy = np.std(xy_errors)
        print(f"  Mean: {mean_xy:.4f} m")
        print(f"  Median: {np.median(xy_errors):.4f} m")
        print(f"  StdDev: {std_xy:.4f} m")
    print("\nHeight (Z) Error:")
    if height_errors:
        mean_z = np.mean(height_errors)
        std_z = np.std(height_errors)
        print(f"  Mean: {mean_z:.4f} m")
        print(f"  Median: {np.median(height_errors):.4f} m")
        print(f"  StdDev: {std_z:.4f} m")
    print("\nCommand Success Rate:")
    print(f"  {success_rate:.2f}% ({sent}/{total_attempts})")
    print(f"  Dropped packets: {dropped}")

    return fig


def main():
    """
    Main function to generate average error analysis chart
    """
    # Request user input for the directory or files
    user_input = input("Please enter the path to the directory containing test result JSON files: ")

    # Check if it's a directory or file
    if os.path.isdir(user_input):
        # It's a directory, load all JSON files
        results = load_all_test_results(user_input)
    elif os.path.isfile(user_input):
        # It's a single file
        results = [load_test_results(user_input)]
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
        print("No results to analyze")
        return

    # Generate output file path
    if os.path.isdir(user_input):
        output_dir = user_input
    else:
        output_dir = os.path.dirname(user_input)

    # Extract RF conditions for filename generation
    combined_data = extract_stable_phase_data(results)
    rf_conditions = combined_data['rf_conditions']

    # Generate filename with RF conditions
    filename = generate_output_filename(rf_conditions, "average_error_analysis")
    output_file = os.path.join(output_dir, filename)

    # Create error analysis chart
    create_average_error_analysis(results, output_file)


if __name__ == "__main__":
    main()
