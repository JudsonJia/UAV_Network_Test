import os
import json
import glob
import math
import numpy as np
from tqdm import tqdm


def distance_to_line_segment(point, line_start, line_end, height_only=False, xy_only=False, vertical_mode="path"):
    """
    Calculate the distance from a point to a line segment

    Args:
        point: Dictionary with x, y, z coordinates
        line_start: Tuple (x, y, z) of line segment start
        line_end: Tuple (x, y, z) of line segment end
        height_only: If True, returns only the height (Z) component error
        xy_only: If True, returns only the XY plane error
        vertical_mode: Mode for height calculations, "path" (along path) or "direct" (direct ascent/descent)

    Returns:
        Distance from point to line segment
    """
    # Extract coordinates
    px, py, pz = point['x'], point['y'], point['z']
    x1, y1, z1 = line_start
    x2, y2, z2 = line_end

    # If line segment length is 0, return distance to endpoint directly
    if (x1 == x2 and y1 == y2 and z1 == z2):
        if height_only:
            return abs(pz - z1)
        elif xy_only:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        else:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2 + (pz - z1) ** 2)

    # Add support for vertical ascent/descent - using direct mode
    if vertical_mode == "direct" and height_only:
        # For vertical movement, use time-based interpolation instead of XY-plane distance
        if 'transition_progress' in point:
            t = point['transition_progress']
            t = max(0.0, min(1.0, t))  # Ensure t is in 0-1 range
        else:
            # If no transition_progress, use default middle value
            t = 0.5

        # Calculate target height based on time progress
        target_z = z1 + t * (z2 - z1)
        return abs(pz - target_z)

    # Standard path calculation (for XY plane or non-vertical 3D path)
    line_length_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if not xy_only:
        line_length_squared += (z2 - z1) ** 2

    if line_length_squared == 0:
        # Prevent division by zero
        if height_only:
            return abs(pz - z1)
        elif xy_only:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        else:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2 + (pz - z1) ** 2)

    # Calculate projection ratio
    if xy_only:
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_squared))
    else:
        t = max(0,
                min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1) + (pz - z1) * (z2 - z1)) / line_length_squared))

    # Calculate projection point coordinates
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    if xy_only:
        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    proj_z = z1 + t * (z2 - z1)

    if height_only:
        return abs(pz - proj_z)

    # Return 3D distance from point to projection point
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2 + (pz - proj_z) ** 2)


def recalculate_trajectory_errors(data):
    """
    Recalculate all trajectory errors in position_data

    Args:
        data: JSON data containing sequence and position_data

    Returns:
        Updated JSON data with corrected errors
    """
    sequence = data.get('sequence', [])
    position_data = data.get('position_data', [])

    if not sequence or not position_data:
        print("Warning: No sequence or position data found")
        return data

    # Detect trajectory type and choose appropriate vertical mode
    vertical_mode = "path"  # Default to path mode

    # First check if vertical mode is already set
    if 'position_accuracy' in data and 'config' in data['position_accuracy']:
        vertical_mode = data['position_accuracy']['config'].get('vertical_mode', "path")

    # Auto-detect vertical trajectory
    if sequence and len(sequence) >= 3:
        # Check if sequence is primarily vertical movement (x,y nearly constant)
        all_x = [p[0] for p in sequence]
        all_y = [p[1] for p in sequence]
        x_variation = max(all_x) - min(all_x)
        y_variation = max(all_y) - min(all_y)

        if x_variation < 0.1 and y_variation < 0.1:
            vertical_mode = "direct"
            print("Detected vertical test sequence, using direct mode for height errors")

    print(f"Using vertical mode: {vertical_mode}")

    # Dictionary to quickly look up waypoints by sequence_index
    waypoints_dict = {}
    for i, waypoint in enumerate(sequence):
        waypoints_dict[i] = waypoint

    # Track updates for statistics
    updates = {
        'total': 0,
        'transit': 0,
        'waypoint': 0,
        'error_diffs': []
    }

    # Process each position data point
    for point in position_data:
        updates['total'] += 1
        old_error = point.get('error', 0)

        # Get target and phase
        phase = point.get('phase', '')
        seq_idx = point.get('sequence_index', 0)

        if phase == 'waypoint':
            updates['waypoint'] += 1
            # For waypoint phase: error is distance to target
            if 'target' in point and 'x' in point and 'y' in point and 'z' in point:
                target = point['target']
                error_xy = math.sqrt((point['x'] - target['x']) ** 2 + (point['y'] - target['y']) ** 2)
                error_z = abs(point['z'] - target['z'])
                error_3d = math.sqrt(error_xy ** 2 + error_z ** 2)

                # Update errors
                point['error'] = error_3d
                point['error_xy'] = error_xy
                point['error_z'] = error_z

                updates['error_diffs'].append(abs(old_error - error_3d))

        elif phase == 'transit':
            updates['transit'] += 1
            # For transit phase: error is distance to ideal path (line segment between waypoints)
            if seq_idx in waypoints_dict:
                target_waypoint = waypoints_dict[seq_idx]

                # Find previous waypoint (ideally should be seq_idx-1, but handle edge cases)
                prev_idx = max(0, seq_idx - 1)
                prev_waypoint = waypoints_dict.get(prev_idx, waypoints_dict[0])

                # Calculate errors using distance to line segment, passing the vertical_mode
                error_3d = distance_to_line_segment(point, prev_waypoint, target_waypoint, vertical_mode=vertical_mode)
                error_xy = distance_to_line_segment(point, prev_waypoint, target_waypoint, xy_only=True,
                                                    vertical_mode=vertical_mode)
                error_z = distance_to_line_segment(point, prev_waypoint, target_waypoint, height_only=True,
                                                   vertical_mode=vertical_mode)

                # Update errors
                point['error'] = error_3d
                point['error_xy'] = error_xy
                point['error_z'] = error_z

                updates['error_diffs'].append(abs(old_error - error_3d))

    # Update data with corrected position_data
    data['position_data'] = position_data

    # Recalculate aggregate metrics in position_accuracy
    if 'position_accuracy' in data:
        # Ensure we save the vertical mode used
        if 'config' not in data['position_accuracy']:
            data['position_accuracy']['config'] = {}
        data['position_accuracy']['config']['vertical_mode'] = vertical_mode

        recalculate_position_accuracy(data)

    # Print update statistics
    avg_diff = np.mean(updates['error_diffs']) if updates['error_diffs'] else 0
    max_diff = max(updates['error_diffs']) if updates['error_diffs'] else 0
    print(f"Updated {updates['total']} points: {updates['transit']} transit, {updates['waypoint']} waypoint")
    print(f"Average error difference: {avg_diff:.4f}m, Max difference: {max_diff:.4f}m")

    return data


def recalculate_position_accuracy(data):
    """
    Recalculate position_accuracy metrics based on updated position_data

    Args:
        data: JSON data containing position_data and position_accuracy
    """
    position_data = data.get('position_data', [])

    if not position_data:
        return

    # Get configuration
    config = data.get('position_accuracy', {}).get('config', {})
    exclude_transit = config.get('exclude_transit', True)

    # Extract errors by phase
    all_errors = [p.get('error', 0) for p in position_data]
    waypoint_errors = [p.get('error', 0) for p in position_data if p.get('phase') == 'waypoint']
    transit_errors = [p.get('error', 0) for p in position_data if p.get('phase') == 'transit']

    # Valid errors (possibly excluding transit phase)
    if exclude_transit:
        valid_errors = waypoint_errors
    else:
        valid_errors = all_errors

    # Extract height errors
    all_height_errors = [p.get('error_z', 0) for p in position_data]
    waypoint_height_errors = [p.get('error_z', 0) for p in position_data if p.get('phase') == 'waypoint']
    transit_height_errors = [p.get('error_z', 0) for p in position_data if p.get('phase') == 'transit']

    # Valid height errors
    if exclude_transit:
        valid_height_errors = waypoint_height_errors
    else:
        valid_height_errors = all_height_errors

    # Extract XY errors
    all_xy_errors = [p.get('error_xy', 0) for p in position_data]
    waypoint_xy_errors = [p.get('error_xy', 0) for p in position_data if p.get('phase') == 'waypoint']
    transit_xy_errors = [p.get('error_xy', 0) for p in position_data if p.get('phase') == 'transit']

    # Valid XY plane errors
    if exclude_transit:
        valid_xy_errors = waypoint_xy_errors
    else:
        valid_xy_errors = all_xy_errors

    # Calculate statistics for each set
    stats = {
        'overall': calculate_error_stats(all_errors),
        'waypoint_phase': calculate_error_stats(waypoint_errors),
        'transit_phase': calculate_error_stats(transit_errors),
        'valid_errors': calculate_error_stats(valid_errors),
        'valid_height': calculate_error_stats(valid_height_errors),
        'valid_xy': calculate_error_stats(valid_xy_errors),
        'height': {
            'overall': calculate_error_stats(all_height_errors),
            'waypoint': calculate_error_stats(waypoint_height_errors),
            'transit': calculate_error_stats(transit_height_errors)
        },
        'xy_plane': {
            'overall': calculate_error_stats(all_xy_errors),
            'waypoint': calculate_error_stats(waypoint_xy_errors),
            'transit': calculate_error_stats(transit_xy_errors)
        },
        'total_points': len(position_data),
        'waypoint_points': len(waypoint_errors),
        'transit_points': len(transit_errors),
        'valid_points': len(valid_errors),
        'excluded_points': len(position_data) - len(valid_errors),
        'config': config
    }

    # Create full position_accuracy structure
    position_accuracy = {
        **stats,
        'average_error': stats['valid_errors']['average'] if stats['valid_errors']['average'] is not None else None,
        'median_error': stats['valid_errors']['median'] if stats['valid_errors']['median'] is not None else None,
        'max_error': stats['valid_errors']['max'] if stats['valid_errors']['max'] is not None else None,
        'min_error': stats['valid_errors']['min'] if stats['valid_errors']['min'] is not None else None,
        'confidence_95': stats['valid_errors']['confidence_95'] if stats['valid_errors'][
                                                                       'confidence_95'] is not None else None,
        'stable_phase': {
            'average': stats['waypoint_phase']['average'],
            'median': stats['waypoint_phase']['median'],
            'max': stats['waypoint_phase']['max'],
            'min': stats['waypoint_phase']['min'],
            'count': len(waypoint_errors),
            'percentage': len(waypoint_errors) / len(position_data) * 100 if position_data else 0,
            'confidence_95': stats['waypoint_phase']['confidence_95']
        }
    }

    # Update data with new position_accuracy
    data['position_accuracy'] = position_accuracy


def calculate_error_stats(error_data):
    """
    Calculate error statistics for a given set of error values

    Args:
        error_data: List of error values

    Returns:
        Dictionary containing error statistics
    """
    if not error_data:
        return {
            'average': None,
            'median': None,
            'min': None,
            'max': None,
            'confidence_95': None
        }

    avg_error = np.mean(error_data)
    median_error = np.median(error_data)
    min_error = min(error_data)
    max_error = max(error_data)

    # Calculate 95% confidence interval
    confidence_95 = None
    if len(error_data) >= 10:
        stdev = np.std(error_data)
        confidence_95 = 1.96 * stdev / math.sqrt(len(error_data))

    return {
        'average': avg_error,
        'median': median_error,
        'min': min_error,
        'max': max_error,
        'confidence_95': confidence_95
    }


def process_file(filepath, output_dir=None, overwrite=False):
    """
    Process a single JSON file, recalculate errors, and save the result

    Args:
        filepath: Path to the JSON file
        output_dir: Directory to save the output (if None, save in same directory)
        overwrite: Whether to overwrite the original file

    Returns:
        Path to the saved output file
    """
    try:
        # Load JSON file
        with open(filepath, 'r') as f:
            data = json.load(f)

        print(f"\nProcessing: {filepath}")

        # Recalculate errors
        updated_data = recalculate_trajectory_errors(data)

        # Determine output path
        if overwrite:
            output_path = filepath
        else:
            filename = os.path.basename(filepath)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"fixed_{filename}")
            else:
                dir_path = os.path.dirname(filepath)
                output_path = os.path.join(dir_path, f"fixed_{filename}")

        # Save updated data
        with open(output_path, 'w') as f:
            json.dump(updated_data, f, indent=2)

        print(f"Successfully saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def process_directory(directory, output_dir=None, overwrite=False):
    """
    Process all JSON files in a directory, recalculating errors

    Args:
        directory: Directory containing JSON files
        output_dir: Directory to save the output files
        overwrite: Whether to overwrite the original files

    Returns:
        List of paths to the saved output files
    """
    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(directory, "*.json"))

    if not json_files:
        print(f"No JSON files found in {directory}")
        return []

    print(f"Found {len(json_files)} JSON files in {directory}")

    # Process each file
    output_files = []
    for filepath in tqdm(json_files, desc="Processing files"):
        output_path = process_file(filepath, output_dir, overwrite)
        if output_path:
            output_files.append(output_path)

    print(f"\nSuccessfully processed {len(output_files)} files")
    return output_files


def main():
    # Get path interactively
    print("\n=== Trajectory Error Recalculation Tool ===")
    path = input("\nEnter path to JSON file or directory containing JSON files: ").strip()

    if not path:
        print("No path entered. Exiting...")
        return

    # Remove quotes if present (in case user copied a path with quotes)
    path = path.strip('"\'')

    # Get output directory option
    output_option = input("\nDo you want to specify an output directory? (y/n): ").strip().lower()
    output_dir = None
    if output_option == 'y':
        output_dir = input("Enter output directory path: ").strip()
        output_dir = output_dir.strip('"\'')  # Remove quotes if present

    # Get overwrite option
    overwrite_option = input("\nDo you want to overwrite original files? (y/n): ").strip().lower()
    overwrite = (overwrite_option == 'y')

    print("\nStarting processing...")

    # Process file or directory
    if os.path.isdir(path):
        output_files = process_directory(path, output_dir, overwrite)
        print(f"\nTotal files processed: {len(output_files)}")
    elif os.path.isfile(path):
        output_path = process_file(path, output_dir, overwrite)
        if output_path:
            print(f"\nFile processed: {output_path}")
    else:
        print(f"\nPath not found: {path}")

    print("\nProcessing complete.")
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()