"""
UAV RF Communication Network Impact Test Script
Directly simulates network conditions at the RF level
Applicable for Crazyflie 2.X with Local Positioning System (LPS)
Includes battery diagnostics - prevents flight if voltage is below 3.8V
"""
import math
import time
import logging
import argparse
import os
import json
import datetime
import random
import queue
import threading
from typing import Dict, List, Optional, Tuple
import statistics
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper
from cflib.utils.reset_estimator import reset_estimator

# Import error calculation functions from fix_trajectory_errors
from fix_trajectory_errors import distance_to_line_segment, calculate_error_stats, recalculate_trajectory_errors

# Configure logging - only save results
os.makedirs('results', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('uav_test')

# Battery voltage threshold
BATTERY_MIN_VOLTAGE = 3.8  # Minimum required voltage in volts


class BatteryDiagnostics:
    """Battery diagnostics and monitoring functions"""

    @staticmethod
    def check_battery_voltage(scf) -> Tuple[float, bool]:
        """
        Check battery voltage and determine if it's safe to fly
        Returns: (voltage, is_safe_to_fly)
        """
        voltage = None
        is_safe = False

        # Create battery log configuration
        battery_log = LogConfig(name='Battery', period_in_ms=100)
        battery_log.add_variable('pm.vbat', 'float')  # Battery voltage

        try:
            # Get current battery voltage
            with SyncLogger(scf, battery_log) as logger_cf:
                for log_entry in logger_cf:
                    data = log_entry[1]
                    voltage = data['pm.vbat']

                    # Check if voltage is above the minimum threshold
                    is_safe = voltage >= BATTERY_MIN_VOLTAGE

                    status_msg = f"Battery voltage: {voltage:.2f}V - {'SAFE' if is_safe else 'LOW'}"
                    print(status_msg)
                    logger.info(status_msg)

                    if not is_safe:
                        warning_msg = f"WARNING: Battery voltage {voltage:.2f}V is below minimum threshold of {BATTERY_MIN_VOLTAGE}V. Flight not permitted."
                        print(warning_msg)
                        logger.warning(warning_msg)

                    break
        except Exception as e:
            error_msg = f"Battery check failed: {e}"
            print(error_msg)
            logger.error(error_msg)

        return voltage, is_safe

    @staticmethod
    def start_battery_monitoring(scf, monitoring_period_ms=1000):
        """
        Start continuous battery monitoring
        Returns a callback removal function to stop monitoring
        """
        battery_data = {
            'voltage': None,
            'is_safe': False,
            'log_config': None,
            'started': False
        }

        # Create battery log configuration
        battery_log = LogConfig(name='BatteryMonitor', period_in_ms=monitoring_period_ms)
        battery_log.add_variable('pm.vbat', 'float')  # Battery voltage

        def battery_callback(timestamp, data, logconf):
            voltage = data['pm.vbat']
            is_safe = voltage >= BATTERY_MIN_VOLTAGE

            # Update status if changed or first reading
            if battery_data['voltage'] != voltage or battery_data['is_safe'] != is_safe:
                status_msg = f"Battery: {voltage:.2f}V - {'SAFE' if is_safe else 'LOW'}"
                print(status_msg)
                logger.info(status_msg)

                # Log warning only when status changes to unsafe
                if battery_data['is_safe'] and not is_safe:
                    warning_msg = f"WARNING: Battery voltage dropped to {voltage:.2f}V (below {BATTERY_MIN_VOLTAGE}V threshold)"
                    print(warning_msg)
                    logger.warning(warning_msg)

            # Update data
            battery_data['voltage'] = voltage
            battery_data['is_safe'] = is_safe

        try:
            battery_log.data_received_cb.add_callback(battery_callback)
            battery_log.start()
            battery_data['log_config'] = battery_log
            battery_data['started'] = True

            logger.info("Battery monitoring started")
        except Exception as e:
            logger.error(f"Failed to start battery monitoring: {e}")

        # Return function to stop monitoring
        def stop_monitoring():
            if battery_data['started']:
                try:
                    battery_log.data_received_cb.remove_callback(battery_callback)
                    battery_log.stop()
                    logger.info("Battery monitoring stopped")
                except Exception as e:
                    logger.error(f"Error stopping battery monitoring: {e}")

        return stop_monitoring


class TrajectoryAwareErrorCalculator3D:
    """Full 3D trajectory-aware error calculator - specially optimized for height processing"""
    # Error calculation constants
    STABILIZATION_THRESHOLD = 0.08  # Threshold for considering the drone in stable phase
    OUTLIER_THRESHOLD = 3.0  # Z-score threshold for outlier detection
    WAYPOINT_PROXIMITY_THRESHOLD = 0.08  # Distance threshold for considering the drone close to a waypoint

    # Height-related parameters
    HEIGHT_WEIGHT = 1.0  # Weight factor for height error (default is 1.0, indicating equal importance to XY plane)
    VERTICAL_TRANSIT_MODE = "path"  # Height transition mode: "path" (along path) or "direct" (direct ascent/descent)

    # New: Transit point processing parameters
    EXCLUDE_TRANSIT_ERRORS = True  # Whether to exclude errors during transition phase
    TRANSIT_ACCELERATION_THRESHOLD = 0.2  # Initial acceleration time ratio in transition phase
    TRANSIT_DECELERATION_THRESHOLD = 0.8  # Final deceleration time ratio in transition phase

    @staticmethod
    def calculate_position_errors(position_data: List[Dict], sequence: List[Tuple[float, float, float]],
                                  height_weight=None, vertical_mode=None, exclude_transit=None) -> Dict:
        """
        Calculate fully 3D trajectory-aware position error metrics, with special attention to height dimension

        Args:
            position_data: List of position data points containing actual and target positions
            sequence: Flight sequence containing waypoints
            height_weight: Optional weight factor for height error (default is None, uses class constant)
            vertical_mode: Optional setting for height transition mode (default is None, uses class constant)
            exclude_transit: Whether to exclude transit point errors (default is None, uses class constant)

        Returns:
            Dictionary containing trajectory-aware error metrics, with additional independent analysis for height dimension
        """
        try:
            # Use provided parameters or default values
            if height_weight is None:
                height_weight = TrajectoryAwareErrorCalculator3D.HEIGHT_WEIGHT

            if vertical_mode is None:
                vertical_mode = TrajectoryAwareErrorCalculator3D.VERTICAL_TRANSIT_MODE

            if exclude_transit is None:
                exclude_transit = TrajectoryAwareErrorCalculator3D.EXCLUDE_TRANSIT_ERRORS

            if not position_data:
                return {
                    'average_error': None,
                    'max_error': None,
                    'median_error': None,
                    'stable_phase': {},
                    'confidence_95': None
                }

            # Error collection lists - overall and separate dimensions
            all_errors = []
            waypoint_errors = []
            transit_errors = []

            # New: Valid errors (filtered according to exclude_transit option)
            valid_errors = []

            # Additional tracking of height dimension errors
            height_errors = {
                'all': [],
                'waypoint': [],
                'transit': []
            }

            # XY plane errors (ignoring height)
            xy_errors = {
                'all': [],
                'waypoint': [],
                'transit': []
            }

            # New: Valid height errors and XY plane errors
            valid_height_errors = []
            valid_xy_errors = []

            # Get thresholds from constants
            STABILIZATION_THRESHOLD = TrajectoryAwareErrorCalculator3D.STABILIZATION_THRESHOLD
            WAYPOINT_PROXIMITY = TrajectoryAwareErrorCalculator3D.WAYPOINT_PROXIMITY_THRESHOLD

            # Group data by sequence index (waypoint target)
            sequence_groups = {}
            for point in position_data:
                if 'sequence_index' not in point or 'target' not in point:
                    continue

                seq_idx = point['sequence_index']
                if seq_idx not in sequence_groups:
                    sequence_groups[seq_idx] = []
                sequence_groups[seq_idx].append(point)

            # Helper function for calculating ideal path between consecutive waypoints
            def distance_to_line_segment(point, line_start, line_end, height_only=False, xy_only=False):
                """
                Calculate the distance from a point to a line segment, optionally only for height or XY plane components

                Args:
                    point: Position point dictionary with x,y,z keys
                    line_start: Line segment start coordinate tuple (x,y,z)
                    line_end: Line segment end coordinate tuple (x,y,z)
                    height_only: If True, returns only the height (Z) component error
                    xy_only: If True, returns only the XY plane error (ignoring height)

                Returns:
                    Distance from point to line segment (complete, height-only, or XY-plane only)
                """
                # Function content remains unchanged...
                # Extract coordinates of point and line segment endpoints
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

                # For vertical movement, use time-based interpolation
                if vertical_mode == "direct" and height_only:
                    if 'transition_progress' in point:
                        t = point['transition_progress']
                        t = max(0.0, min(1.0, t))
                    else:
                        t = 0.5

                    target_z = z1 + t * (z2 - z1)
                    return abs(pz - target_z)

                # Standard path calculation (for XY plane or 3D path)
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
                    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1) + (pz - z1) * (
                            z2 - z1)) / line_length_squared))

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

            # Find adjacent waypoints to calculate transition paths
            prev_waypoint = None
            sequence_waypoints = {}
            base_x, base_y, base_z = 0, 0, 0  # Base coordinate offset

            # Extract base offset from first data point (assuming all data points use the same base offset)
            if position_data and 'target' in position_data[0]:
                first_target = position_data[0]['target']
                first_seq_idx = position_data[0]['sequence_index']
                if 0 <= first_seq_idx < len(sequence):
                    base_x = first_target['x'] - sequence[first_seq_idx][0]
                    base_y = first_target['y'] - sequence[first_seq_idx][1]
                    base_z = first_target['z'] - sequence[first_seq_idx][2]

            # Record start time for each sequence, used to calculate transition progress
            sequence_start_times = {}
            sequence_durations = {}

            # Process each waypoint group
            for seq_idx in sorted(sequence_groups.keys()):
                points = sequence_groups[seq_idx]
                if not points:
                    continue

                # Record sequence start time
                sequence_start_times[seq_idx] = min(point['time'] for point in points)

                # If there is a next sequence, estimate transition duration
                if seq_idx + 1 in sequence_groups:
                    next_start = min(point['time'] for point in sequence_groups[seq_idx + 1])
                    sequence_durations[seq_idx] = next_start - sequence_start_times[seq_idx]

                current_target = points[0]['target']
                current_waypoint = (current_target['x'], current_target['y'], current_target['z'])
                sequence_waypoints[seq_idx] = current_waypoint

                # Process points for each waypoint
                for point in points:
                    # Calculate error to current target
                    error_x = abs(point['x'] - point['target']['x'])
                    error_y = abs(point['y'] - point['target']['y'])
                    error_z = abs(point['z'] - point['target']['z'])

                    # Plane error and 3D error
                    xy_error = math.sqrt(error_x ** 2 + error_y ** 2)
                    total_error_to_target = math.sqrt(xy_error ** 2 + error_z ** 2)

                    # Calculate transition progress in this sequence (for judging acceleration and deceleration phases)
                    transition_progress = 0.5  # Default middle value
                    if seq_idx in sequence_start_times and seq_idx in sequence_durations:
                        elapsed = point['time'] - sequence_start_times[seq_idx]
                        if sequence_durations[seq_idx] > 0:
                            transition_progress = elapsed / sequence_durations[seq_idx]

                    # Record transition progress
                    point['transition_progress'] = transition_progress

                    # Determine if point is in acceleration or deceleration phase
                    is_acceleration_phase = transition_progress < TrajectoryAwareErrorCalculator3D.TRANSIT_ACCELERATION_THRESHOLD
                    is_deceleration_phase = transition_progress > TrajectoryAwareErrorCalculator3D.TRANSIT_DECELERATION_THRESHOLD

                    # Determine if point is in waypoint phase or transit phase
                    if total_error_to_target < WAYPOINT_PROXIMITY:
                        # We're near waypoint - consider as waypoint phase
                        point['phase'] = 'waypoint'

                        # Store errors for various dimensions
                        point['error'] = total_error_to_target
                        point['error_xy'] = xy_error
                        point['error_z'] = error_z

                        # Add to appropriate error lists
                        waypoint_errors.append(total_error_to_target)
                        all_errors.append(total_error_to_target)

                        # Check if this point should be excluded (deceleration phase)
                        if not (exclude_transit and is_deceleration_phase):
                            valid_errors.append(total_error_to_target)
                            valid_height_errors.append(error_z)
                            valid_xy_errors.append(xy_error)

                        height_errors['all'].append(error_z)
                        height_errors['waypoint'].append(error_z)

                        xy_errors['all'].append(xy_error)
                        xy_errors['waypoint'].append(xy_error)

                        # Check if stable
                        if (error_x < STABILIZATION_THRESHOLD and
                                error_y < STABILIZATION_THRESHOLD and
                                error_z < STABILIZATION_THRESHOLD):
                            point['stabilized'] = True
                        else:
                            point['stabilized'] = False
                    else:
                        # We're transitioning between waypoints - calculate deviation from ideal path
                        point['phase'] = 'transit'
                        point['stabilized'] = False

                        # Find previous and current waypoints to define ideal path
                        if prev_waypoint is not None and current_waypoint is not None:
                            # Calculate full 3D distance to ideal path
                            path_deviation_3d = distance_to_line_segment(point, prev_waypoint, current_waypoint)

                            # Calculate height and XY plane deviations separately
                            path_deviation_z = distance_to_line_segment(point, prev_waypoint, current_waypoint,
                                                                        height_only=True)
                            path_deviation_xy = distance_to_line_segment(point, prev_waypoint, current_waypoint,
                                                                         xy_only=True)

                            # Store errors for various dimensions
                            point['error'] = path_deviation_3d
                            point['error_xy'] = path_deviation_xy
                            point['error_z'] = path_deviation_z

                            # Add to appropriate error lists
                            transit_errors.append(path_deviation_3d)
                            all_errors.append(path_deviation_3d)

                            # New: If not excluding transit errors, add to valid error lists
                            if not exclude_transit:
                                valid_errors.append(path_deviation_3d)
                                valid_height_errors.append(path_deviation_z)
                                valid_xy_errors.append(path_deviation_xy)

                            height_errors['all'].append(path_deviation_z)
                            height_errors['transit'].append(path_deviation_z)

                            xy_errors['all'].append(path_deviation_xy)
                            xy_errors['transit'].append(path_deviation_xy)
                        else:
                            # If no previous waypoint (this is the first waypoint), use error to target
                            point['error'] = total_error_to_target
                            point['error_xy'] = xy_error
                            point['error_z'] = error_z

                            transit_errors.append(total_error_to_target)
                            all_errors.append(total_error_to_target)

                            # New: If not excluding transit errors, add to valid error lists
                            if not exclude_transit:
                                valid_errors.append(total_error_to_target)
                                valid_height_errors.append(error_z)
                                valid_xy_errors.append(xy_error)

                            height_errors['all'].append(error_z)
                            height_errors['transit'].append(error_z)

                            xy_errors['all'].append(xy_error)
                            xy_errors['transit'].append(xy_error)

                # Update previous waypoint for next sequence group
                prev_waypoint = current_waypoint

            # Filter outliers
            filtered_all_errors = TrajectoryAwareErrorCalculator3D._filter_outliers(all_errors)
            filtered_waypoint_errors = TrajectoryAwareErrorCalculator3D._filter_outliers(waypoint_errors)
            filtered_transit_errors = TrajectoryAwareErrorCalculator3D._filter_outliers(transit_errors)
            filtered_valid_errors = TrajectoryAwareErrorCalculator3D._filter_outliers(valid_errors)
            filtered_valid_height_errors = TrajectoryAwareErrorCalculator3D._filter_outliers(valid_height_errors)
            filtered_valid_xy_errors = TrajectoryAwareErrorCalculator3D._filter_outliers(valid_xy_errors)

            # Filter outliers from height errors
            filtered_height_errors = {
                'all': TrajectoryAwareErrorCalculator3D._filter_outliers(height_errors['all']),
                'waypoint': TrajectoryAwareErrorCalculator3D._filter_outliers(height_errors['waypoint']),
                'transit': TrajectoryAwareErrorCalculator3D._filter_outliers(height_errors['transit'])
            }

            # Filter outliers from XY plane errors
            filtered_xy_errors = {
                'all': TrajectoryAwareErrorCalculator3D._filter_outliers(xy_errors['all']),
                'waypoint': TrajectoryAwareErrorCalculator3D._filter_outliers(xy_errors['waypoint']),
                'transit': TrajectoryAwareErrorCalculator3D._filter_outliers(xy_errors['transit'])
            }

            # Calculate overall statistics
            result = {
                'overall': TrajectoryAwareErrorCalculator3D._calculate_error_stats(filtered_all_errors),
                'waypoint_phase': TrajectoryAwareErrorCalculator3D._calculate_error_stats(filtered_waypoint_errors),
                'transit_phase': TrajectoryAwareErrorCalculator3D._calculate_error_stats(filtered_transit_errors),
                'valid_errors': TrajectoryAwareErrorCalculator3D._calculate_error_stats(filtered_valid_errors),
                'valid_height': TrajectoryAwareErrorCalculator3D._calculate_error_stats(filtered_valid_height_errors),
                'valid_xy': TrajectoryAwareErrorCalculator3D._calculate_error_stats(filtered_valid_xy_errors),

                # Separate analysis for height dimension
                'height': {
                    'overall': TrajectoryAwareErrorCalculator3D._calculate_error_stats(filtered_height_errors['all']),
                    'waypoint': TrajectoryAwareErrorCalculator3D._calculate_error_stats(
                        filtered_height_errors['waypoint']),
                    'transit': TrajectoryAwareErrorCalculator3D._calculate_error_stats(
                        filtered_height_errors['transit'])
                },

                # Separate analysis for XY plane
                'xy_plane': {
                    'overall': TrajectoryAwareErrorCalculator3D._calculate_error_stats(filtered_xy_errors['all']),
                    'waypoint': TrajectoryAwareErrorCalculator3D._calculate_error_stats(filtered_xy_errors['waypoint']),
                    'transit': TrajectoryAwareErrorCalculator3D._calculate_error_stats(filtered_xy_errors['transit'])
                },

                'total_points': len(all_errors),
                'waypoint_points': len(waypoint_errors),
                'transit_points': len(transit_errors),
                'valid_points': len(valid_errors),
                'excluded_points': len(all_errors) - len(valid_errors),
                'outliers_removed': {
                    'overall': len(all_errors) - len(filtered_all_errors),
                    'waypoint': len(waypoint_errors) - len(filtered_waypoint_errors),
                    'transit': len(transit_errors) - len(filtered_transit_errors),
                    'valid': len(valid_errors) - len(filtered_valid_errors),
                    'height': {
                        'overall': len(height_errors['all']) - len(filtered_height_errors['all']),
                        'waypoint': len(height_errors['waypoint']) - len(filtered_height_errors['waypoint']),
                        'transit': len(height_errors['transit']) - len(filtered_height_errors['transit'])
                    },
                    'xy_plane': {
                        'overall': len(xy_errors['all']) - len(filtered_xy_errors['all']),
                        'waypoint': len(xy_errors['waypoint']) - len(filtered_xy_errors['waypoint']),
                        'transit': len(xy_errors['transit']) - len(filtered_xy_errors['transit'])
                    }
                },
                'config': {
                    'height_weight': height_weight,
                    'vertical_mode': vertical_mode,
                    'exclude_transit': exclude_transit
                }
            }

            if result['waypoint_phase']['average'] is not None:
                if exclude_transit and result['valid_errors']['average'] is not None:
                    # Use valid errors (excluding transit points)
                    result['average_error'] = result['valid_errors']['average']
                    result['median_error'] = result['valid_errors']['median']
                    result['max_error'] = result['valid_errors']['max']
                    result['min_error'] = result['valid_errors']['min']
                    result['confidence_95'] = result['valid_errors']['confidence_95']
                else:
                    # Use all errors
                    result['average_error'] = result['overall']['average']
                    result['median_error'] = result['overall']['median']
                    result['max_error'] = result['overall']['max']
                    result['min_error'] = result['overall']['min']
                    result['confidence_95'] = result['overall']['confidence_95']

                result['stable_phase'] = {
                    'average': result['waypoint_phase']['average'],
                    'median': result['waypoint_phase']['median'],
                    'max': result['waypoint_phase']['max'],
                    'min': result['waypoint_phase']['min'],
                    'count': len(filtered_waypoint_errors),
                    'percentage': len(filtered_waypoint_errors) / len(all_errors) * 100 if all_errors else 0,
                    'confidence_95': result['waypoint_phase']['confidence_95']
                }

            return result

        except Exception as e:
            print(f"Error calculating position errors: {e}")
            import traceback
            traceback.print_exc()
            return {
                'average_error': None,
                'max_error': None,
                'median_error': None,
                'stable_phase': {},
                'confidence_95': None,
                'calculation_error': str(e)
            }

    @staticmethod
    def _calculate_error_stats(error_data: List[float]) -> Dict:
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

        avg_error = statistics.mean(error_data)
        median_error = statistics.median(error_data)
        min_error = min(error_data)
        max_error = max(error_data)

        # Calculate 95% confidence interval
        confidence_95 = None
        if len(error_data) >= 10:
            stdev = statistics.stdev(error_data)
            confidence_95 = 1.96 * stdev / math.sqrt(len(error_data))

        return {
            'average': avg_error,
            'median': median_error,
            'min': min_error,
            'max': max_error,
            'confidence_95': confidence_95
        }

    @staticmethod
    def _filter_outliers(data_points: List[float]) -> List[float]:
        """
        Filter outliers using z-score method

        Args:
            data_points: List of error values

        Returns:
            Filtered list with outliers removed
        """
        try:
            if not data_points or len(data_points) < 4:  # Need minimum number of points for meaningful statistics
                return data_points

            # Calculate mean and standard deviation
            mean = statistics.mean(data_points)
            stdev = statistics.stdev(data_points)

            if stdev == 0:  # All points are the same value
                return data_points

            # Filter out points with z-score greater than threshold
            filtered = [x for x in data_points if
                        abs((x - mean) / stdev) < TrajectoryAwareErrorCalculator3D.OUTLIER_THRESHOLD]

            return filtered
        except Exception as e:
            print(f"Error filtering outliers: {e}")
            # Return original data in case of error
            return data_points


class RFNetworkSimulator:
    """RF Network Simulator - Directly controls command transmission behavior"""

    def __init__(self, bandwidth_kbps: Optional[float] = None,
                 latency_ms: Optional[float] = None,
                 packet_loss_rate: Optional[float] = None):
        self.bandwidth_kbps = bandwidth_kbps
        self.latency_ms = latency_ms
        self.packet_loss_rate = packet_loss_rate

        # Bandwidth control related
        self.last_send_time = time.time()
        self.bytes_sent = 0
        self.byte_window = []

        # Delay control related
        self.delayed_commands = queue.Queue()
        self._start_delay_worker()

        logger.info(
            f"RF Network Simulator initialized: Bandwidth={bandwidth_kbps}Kbps, Latency={latency_ms}ms, Packet Loss Rate={packet_loss_rate}%")

    def _start_delay_worker(self):
        """Start delay processing worker thread"""

        def delay_worker():
            while True:
                try:
                    send_time, command, cf = self.delayed_commands.get(timeout=0.1)
                    current_time = time.time()
                    if current_time >= send_time:
                        # Execute delayed command
                        command()
                    else:
                        # Still need to wait, put back in queue
                        self.delayed_commands.put((send_time, command, cf))
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Delay processing thread error: {e}")

        worker_thread = threading.Thread(target=delay_worker, daemon=True)
        worker_thread.start()

    def send_position_with_conditions(self, cf, x: float, y: float, z: float, yaw: float) -> bool:
        """
        Send position command according to network conditions
        Returns: True if command was sent, False if dropped
        """
        # 1. Check packet loss
        if self.packet_loss_rate and random.random() < self.packet_loss_rate / 100:
            logger.debug(f"Simulated packet loss: Position command ({x:.3f}, {y:.3f}, {z:.3f}) dropped")
            return False

        # 2. Calculate command size (estimate)
        command_size_bytes = 16  # Estimated size of a position command

        # 3. Bandwidth limitation
        if self.bandwidth_kbps:
            current_time = time.time()

            # Clean up old send records
            cutoff_time = current_time - 1.0  # Keep records within 1 second
            self.byte_window = [(t, b) for t, b in self.byte_window if t > cutoff_time]

            # Calculate current traffic in 1 second
            current_bytes_in_window = sum(b for _, b in self.byte_window)
            max_bytes_per_second = self.bandwidth_kbps * 1024 / 8

            if current_bytes_in_window + command_size_bytes > max_bytes_per_second:
                # Calculate required wait time
                excess_bytes = (current_bytes_in_window + command_size_bytes) - max_bytes_per_second
                wait_time = excess_bytes / (max_bytes_per_second)
                logger.debug(f"Bandwidth limit: Waiting {wait_time * 1000:.1f}ms")
                time.sleep(wait_time)

            # Record transmission
            self.byte_window.append((time.time(), command_size_bytes))

        # 4. Delay processing
        if self.latency_ms:
            delay_seconds = self.latency_ms / 1000
            send_time = time.time() + delay_seconds

            def delayed_send():
                cf.commander.send_position_setpoint(x, y, z, yaw)
                logger.debug(f"Delayed position command sent: ({x:.3f}, {y:.3f}, {z:.3f}) delay={self.latency_ms}ms")

            self.delayed_commands.put((send_time, delayed_send, cf))
        else:
            # No delay, send directly
            cf.commander.send_position_setpoint(x, y, z, yaw)

        return True


def set_initial_position(scf, x, y, z, yaw_deg):
    """Set initial position and yaw angle"""
    scf.cf.param.set_value('kalman.initialX', x)
    scf.cf.param.set_value('kalman.initialY', y)
    scf.cf.param.set_value('kalman.initialZ', z)

    yaw_radians = math.radians(yaw_deg)
    scf.cf.param.set_value('kalman.initialYaw', yaw_radians)


def wait_for_position_estimator(scf):
    """Wait for positioning system initialization"""
    print('Waiting for positioning system initialization...')
    logger.info('Waiting for positioning system initialization...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=100)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_x_history = [1000] * 10
    var_y_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.005

    with SyncLogger(scf, log_config) as logger_cf:
        count = 0
        for log_entry in logger_cf:
            data = log_entry[1]

            x = data['kalman.varPX']
            y = data['kalman.varPY']
            z = data['kalman.varPZ']

            var_x_history.append(x)
            var_y_history.append(y)
            var_z_history.append(z)

            var_x_history.pop(0)
            var_y_history.pop(0)
            var_z_history.pop(0)

            if (max(var_x_history) < threshold and
                    max(var_y_history) < threshold and
                    max(var_z_history) < threshold):
                print('Positioning system initialization complete')
                logger.info('Positioning system initialization complete')
                break

            count += 1
            if count % 10 == 0:
                status_msg = f'Waiting for positioning... X: {x:.4f}, Y: {y:.4f}, Z: {z:.4f}'
                print(status_msg)
                logger.info(status_msg)

            if count > 150:
                warning_msg = 'Warning: Positioning system not fully initialized within expected time'
                print(warning_msg)
                logger.warning(warning_msg)
                break


def run_sequence_with_rf_simulation(scf, sequence: List[Tuple[float, float, float]],
                                    rf_sim: RFNetworkSimulator,
                                    base_x=0.0, base_y=0.0, base_z=0.0, yaw=0):
    """Execute flight sequence - using RF network simulation with improved control logic"""
    cf = scf.cf

    # Check battery before flight
    voltage, is_safe = BatteryDiagnostics.check_battery_voltage(scf)

    # Exit if battery is unsafe
    if not is_safe:
        abort_msg = f"FLIGHT ABORTED: Battery voltage ({voltage:.2f}V) below minimum threshold ({BATTERY_MIN_VOLTAGE}V)"
        print(abort_msg)
        logger.error(abort_msg)
        return {
            'error': 'low_battery',
            'battery_voltage': voltage,
            'battery_required': BATTERY_MIN_VOLTAGE
        }

    # Start battery monitoring
    stop_battery_monitoring = BatteryDiagnostics.start_battery_monitoring(scf)

    # Record performance data
    response_time = None
    first_command_time = None
    position_data = []
    command_stats = {
        'sent': 0,
        'dropped': 0,
        'total_attempts': 0
    }

    # Position tracking variables
    last_position_update_time = 0
    position_data_lock = threading.Lock()
    current_height = 0.0
    current_x = 0.0
    current_y = 0.0

    # Unlock Crazyflie
    cf.platform.send_arming_request(True)
    time.sleep(1.0)

    # Create position log configuration - increased update rate
    position_log = LogConfig(name='Position', period_in_ms=50)  # 50ms = 20Hz update rate
    position_log.add_variable('kalman.stateX', 'float')
    position_log.add_variable('kalman.stateY', 'float')
    position_log.add_variable('kalman.stateZ', 'float')

    # Get current height
    try:
        with SyncLogger(scf, position_log) as logger_cf:
            for log_entry in logger_cf:
                data = log_entry[1]
                current_height = data['kalman.stateZ']
                current_x = data['kalman.stateX']
                current_y = data['kalman.stateY']
                last_position_update_time = time.time()
                pos_msg = f"Initial position: X: {current_x:.3f}, Y: {current_y:.3f}, Z: {current_height:.3f}"
                print(pos_msg)
                logger.info(pos_msg)
                break
    except Exception as e:
        print(f"Initial position measurement failed: {e}")
        logger.error(f"Initial position measurement failed: {e}")
        current_height = 0.0
        current_x = 0.0
        current_y = 0.0

    try:
        # Execute each position in the sequence
        for position_idx, position in enumerate(sequence):
            print(f'Target position {position}')
            logger.info(f'Target position {position}')

            x = position[0] + base_x
            y = position[1] + base_y
            z = position[2] + base_z

            # Determine if this is a takeoff command
            is_takeoff_command = z > current_height + 0.1

            # Record takeoff command time
            if is_takeoff_command and first_command_time is None:
                cmd_msg = f"Recording takeoff command time (Current height: {current_height:.3f}m, Target height: {z:.3f}m)"
                print(cmd_msg)
                logger.info(cmd_msg)
                first_command_time = time.time()

            # Calculate distance to target to adjust flight parameters
            delta_x = abs(x - current_x)
            delta_y = abs(y - current_y)
            delta_z = abs(z - current_height)
            total_distance = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

            # Adjust arrival threshold and stabilization time based on distance
            if total_distance > 0.25:
                tolerance = 0.08
                stable_steps = 40  # Increased for better stability
            else:
                tolerance = 0.08
                stable_steps = 20

            # Enhanced position callback with thread safety
            def position_callback(timestamp, data, logconf):
                nonlocal response_time, current_height, current_x, current_y, last_position_update_time

                with position_data_lock:
                    # Record position data
                    pos_data = {
                        'x': data['kalman.stateX'],
                        'y': data['kalman.stateY'],
                        'z': data['kalman.stateZ'],
                        'time': time.time(),
                        'target': {'x': x, 'y': y, 'z': z},
                        'position_index': len(position_data),
                        'sequence_index': position_idx
                    }
                    position_data.append(pos_data)

                    # Update current position
                    current_height = data['kalman.stateZ']
                    current_x = data['kalman.stateX']
                    current_y = data['kalman.stateY']
                    last_position_update_time = time.time()

                # Print position every 5 data points
                if len(position_data) % 5 == 0:
                    pos_msg = f"Current position: X: {current_x:.3f}, Y: {current_y:.3f}, Z: {current_height:.3f}"
                    print(pos_msg)
                    logger.info(pos_msg)

                # Detect response time - first significant height change
                if response_time is None and data['kalman.stateZ'] > 0.1 and first_command_time is not None:
                    response_time = time.time() - first_command_time
                    resp_msg = f"Command response time: {response_time:.3f} seconds"
                    print(resp_msg)
                    logger.info(resp_msg)

            try:
                position_log.data_received_cb.add_callback(position_callback)
                position_log.start()

                # Control loop variables
                start_time = time.time()
                position_reached = False
                stable_count = 0
                max_time = 10.0  # Maximum 10 seconds per waypoint

                # Control frequency management
                control_hz = 100  # Standard 100Hz Crazyflie position control frequency
                control_interval = 1.0 / control_hz
                next_control_time = time.time()
                position_timeout = 0.5  # 500ms timeout for position updates

                # Improved control loop
                while time.time() - start_time < max_time:
                    current_time = time.time()

                    # Check for position data timeout
                    if current_time - last_position_update_time > position_timeout:
                        warning_msg = f"Position data timeout detected! Last update was {current_time - last_position_update_time:.2f}s ago"
                        print(warning_msg)
                        logger.warning(warning_msg)

                    # Only execute at control frequency
                    if current_time >= next_control_time:
                        with position_data_lock:
                            # Check stability based on current position
                            if position_data:
                                last_pos = position_data[-1]
                                error_x = abs(last_pos['x'] - x)
                                error_y = abs(last_pos['y'] - y)
                                error_z = abs(last_pos['z'] - z)
                                total_error = math.sqrt(error_x ** 2 + error_y ** 2 + error_z ** 2)

                                # Improved stability detection logic
                                if position_reached:
                                    # Already in vicinity, checking for stability
                                    if total_error < tolerance:
                                        stable_count += 1
                                        if stable_count >= stable_steps:
                                            logger.info(
                                                f"Position stable for {stable_count} steps (error={total_error:.3f}m), target complete")
                                            break
                                    else:
                                        # Allow some instability but don't reset completely
                                        stable_count = max(0, stable_count - 1)
                                else:
                                    # Not yet reached, check for proximity
                                    if total_error < 0.15:
                                        position_reached = True
                                        logger.info(
                                            f"Approaching target position (error={total_error:.3f}m), entering stabilization phase")

                        # Send position command
                        command_stats['total_attempts'] += 1
                        success = rf_sim.send_position_with_conditions(cf, x, y, z, yaw)

                        if success:
                            command_stats['sent'] += 1
                        else:
                            command_stats['dropped'] += 1

                        # Calculate next control time
                        next_control_time += control_interval

                        # Reset if we've fallen behind too much
                        if current_time > next_control_time + control_interval:
                            next_control_time = current_time + control_interval
                            logger.warning("Control timing drift detected - resynchronizing")

                        # Log status periodically
                        if command_stats['total_attempts'] % 50 == 0 and position_data:
                            with position_data_lock:
                                if position_data:
                                    last_pos = position_data[-1]
                                    error_x = abs(last_pos['x'] - x)
                                    error_y = abs(last_pos['y'] - y)
                                    error_z = abs(last_pos['z'] - z)
                                    total_error = math.sqrt(error_x ** 2 + error_y ** 2 + error_z ** 2)
                                    step_msg = f"Command #{command_stats['total_attempts']}: Error={total_error:.3f}m, Stable={stable_count}/{stable_steps}"
                                    print(step_msg)
                                    logger.info(step_msg)

                    # Smart sleep to maintain control frequency
                    sleep_time = min(0.001, max(0, next_control_time - time.time()))
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                # Remove callback
                position_log.data_received_cb.remove_callback(position_callback)

                # Print position arrival status
                with position_data_lock:
                    if position_data:
                        last_pos = position_data[-1]
                        final_error_x = abs(last_pos['x'] - x)
                        final_error_y = abs(last_pos['y'] - y)
                        final_error_z = abs(last_pos['z'] - z)
                        final_total_error = math.sqrt(final_error_x ** 2 + final_error_y ** 2 + final_error_z ** 2)

                        status_msg = f"Position {position_idx + 1} complete, Final error: {final_total_error:.3f}m (X={final_error_x:.3f}, Y={final_error_y:.3f}, Z={final_error_z:.3f})"
                        print(status_msg)
                        logger.info(status_msg)

                # Additional stabilization wait time before next waypoint
                time.sleep(0.5)

            except Exception as e:
                print(f"Position data collection error: {e}")
                logger.error(f"Position data collection error: {e}")
                try:
                    position_log.data_received_cb.remove_callback(position_callback)
                except:
                    pass

        # Stop log configuration
        try:
            position_log.stop()
        except:
            pass

        # Stop sending setpoints
        cf.commander.send_stop_setpoint()
        cf.commander.send_notify_setpoint_stop()

        # Lock UAV
        cf.platform.send_arming_request(False)

        # Wait for delayed commands to complete
        time.sleep(1.0)

        completion_msg = "Sequence test complete"
        print(completion_msg)
        logger.info(completion_msg)

        # Print command statistics
        print(f"\nCommand statistics:")
        print(f"Attempt to send: {command_stats['total_attempts']}")
        print(f"Successfully sent: {command_stats['sent']}")
        print(f"Dropped: {command_stats['dropped']}")
        print(f"Success rate: {command_stats['sent'] / command_stats['total_attempts'] * 100:.1f}%")

        # Create appropriate data structure for error calculation
        # Convert position_data and sequence into format expected by recalculate_trajectory_errors
        formatted_data = {
            'sequence': [(p[0], p[1], p[2]) for p in sequence],
            'position_data': position_data
        }

        # NEW: Use imported error calculation methods instead of TrajectoryAwareErrorCalculator3D
        updated_data = recalculate_trajectory_errors(formatted_data)
        error_metrics = updated_data.get('position_accuracy', {})

        # Print error metrics
        print_3d_trajectory_error_metrics(error_metrics)

        # save results
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        result_summary = {
            'timestamp': timestamp,
            'response_time': response_time,
            'sequence': sequence,
            'position_data': position_data,
            'command_stats': command_stats,
            'first_command_time': first_command_time,
            'rf_conditions': {
                'bandwidth_kbps': rf_sim.bandwidth_kbps,
                'latency_ms': rf_sim.latency_ms,
                'packet_loss_rate': rf_sim.packet_loss_rate
            },
            'position_accuracy': error_metrics,
            'error_calculation': {
                'method': 'fix_trajectory_errors',
            },
            'battery': {
                'start_voltage': voltage,
                'minimum_required': BATTERY_MIN_VOLTAGE
            }
        }

        result_file = f"{results_dir}/rf_test_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(result_summary, f, indent=2)

        save_msg = f"Test results saved to: {result_file}"
        print(save_msg)
        logger.info(save_msg)

        return {
            'response_time': response_time,
            'command_stats': command_stats,
            'position_accuracy': error_metrics,
            'result_file': result_file,
            'battery_voltage': voltage
        }

    finally:
        # Always stop battery monitoring
        try:
            stop_battery_monitoring()
        except:
            pass

        # Ensure logs are properly stopped and the drone is safely landed
        try:
            position_log.stop()
            cf.commander.send_stop_setpoint()
            cf.platform.send_arming_request(False)
        except:
            pass


def run_vertical_sequence_with_rf_simulation(scf, rf_sim, base_x=0.0, base_y=0.0, base_z=0.0, yaw=0):
    """Execute vertical ascent/descent sequence - 0.3m → 0.6m → 0.9m → 0.6m → 0.3m
    using "direct" mode
    """

    # Define vertical sequence with more gradual height changes
    vertical_sequence = [
        (0, 0, 0.3),  # Take off to 0.3m
        (0, 0, 0.6),  # Rise to 0.6m
        (0, 0, 0.9),  # Rise to 0.9m (reduced from 1.2m to 0.9m for more stability)
        (0, 0, 0.6),  # Descend to 0.6m
        (0, 0, 0.3),  # Descend back to 0.3m for safe landing
    ]

    logger.info(f"Vertical sequence: {vertical_sequence}")

    # Use existing sequence execution function
    return run_sequence_with_rf_simulation(scf, vertical_sequence, rf_sim, base_x, base_y, base_z, yaw)


def run_square_trajectory_with_rf_simulation(scf, rf_sim, side_length=1.0, height=0.6,
                                             base_x=0.0, base_y=0.0, base_z=0.0, yaw=0):
    """Execute simplified square trajectory test with 5 waypoints
    using "path" mode
    """

    # Generate simplified square waypoints - only 4 corners + starting point
    square_sequence = [
        (0, 0, height),  # Starting point (center)
        (side_length / 2, side_length / 2, height),  # Top right corner
        (-side_length / 2, side_length / 2, height),  # Top left corner
        (-side_length / 2, -side_length / 2, height),  # Bottom left corner
        (side_length / 2, -side_length / 2, height),  # Bottom right corner
        (0, 0, height),  # Return to starting point
    ]

    logger.info(f"Square trajectory: Side length={side_length}m, Height={height}m")

    # Use existing sequence execution function
    return run_sequence_with_rf_simulation(scf, square_sequence, rf_sim, base_x, base_y, base_z, yaw)


def optimize_controller_for_height(scf):
    """Optimize control parameters to enhance height stability"""
    try:
        # Adjust PID parameters for high altitude and descent process
        # Reduce P gain, increase I and D to improve stability
        scf.cf.param.set_value('posCtlPid.zKp', 1.5)  # Reduce P gain to decrease oscillation
        scf.cf.param.set_value('posCtlPid.zKi', 0.6)  # Increase I gain to improve steady-state accuracy
        scf.cf.param.set_value('posCtlPid.zKd', 0.4)  # Increase D gain to suppress overshoot

        scf.cf.param.set_value('velCtlPid.vzKd', 0.5)  # Increase velocity control damping

        logger.info("Successfully set position control parameters for improved stability")
    except Exception as e:
        logger.error(f"Error setting controller parameters: {e}")
        print(f"Unable to set controller parameters: {e}")


def print_3d_trajectory_error_metrics(error_metrics):
    """
    Print complete 3D trajectory-aware error metrics, including independent analysis of height dimension

    Args:
        error_metrics: Error metrics dictionary from TrajectoryAwareErrorCalculator3D
    """
    print("\n==== Complete 3D Trajectory Analysis ====")

    # Configuration information
    if 'config' in error_metrics:
        config = error_metrics['config']
        print(f"\n⚙️ Analysis Configuration:")
        print(f"  Height Weight: {config['height_weight']}")
        print(
            f"  Vertical Transit Mode: {config['vertical_mode']} ({'Along Path' if config['vertical_mode'] == 'path' else 'Direct Ascent/Descent'})")

    # Overall 3D statistics
    print("\n🔍 Overall 3D Accuracy (Entire Flight):")
    if error_metrics['overall']['average'] is not None:
        print(f"  Average 3D Error: {error_metrics['overall']['average']:.4f} m")
        print(f"  Median 3D Error: {error_metrics['overall']['median']:.4f} m")
        print(f"  Minimum 3D Error: {error_metrics['overall']['min']:.4f} m")
        print(f"  Maximum 3D Error: {error_metrics['overall']['max']:.4f} m")

        if error_metrics['overall']['confidence_95'] is not None:
            print(f"  95% Confidence Interval: ±{error_metrics['overall']['confidence_95']:.4f} m")
    else:
        print("  No overall data available")

    # XY plane analysis
    print("\n📏 Horizontal Plane (XY) Analysis:")
    if error_metrics['xy_plane']['overall']['average'] is not None:
        print(f"  Average Horizontal Error: {error_metrics['xy_plane']['overall']['average']:.4f} m")
        print(f"  Median Horizontal Error: {error_metrics['xy_plane']['overall']['median']:.4f} m")
        print(f"  Minimum Horizontal Error: {error_metrics['xy_plane']['overall']['min']:.4f} m")
        print(f"  Maximum Horizontal Error: {error_metrics['xy_plane']['overall']['max']:.4f} m")

    # Height analysis
    print("\n📐 Height (Z) Analysis:")
    if error_metrics['height']['overall']['average'] is not None:
        print(f"  Average Height Error: {error_metrics['height']['overall']['average']:.4f} m")
        print(f"  Median Height Error: {error_metrics['height']['overall']['median']:.4f} m")
        print(f"  Minimum Height Error: {error_metrics['height']['overall']['min']:.4f} m")
        print(f"  Maximum Height Error: {error_metrics['height']['overall']['max']:.4f} m")

    # Waypoint phase statistics
    print("\n🎯 Waypoint Phase Accuracy (When Drone is Near Target Points):")
    if error_metrics['waypoint_phase']['average'] is not None:
        print(f"  Average 3D Error: {error_metrics['waypoint_phase']['average']:.4f} m")
        print(f"  Average Horizontal Error: {error_metrics['xy_plane']['waypoint']['average']:.4f} m")
        print(f"  Average Height Error: {error_metrics['height']['waypoint']['average']:.4f} m")

        waypoint_percentage = (error_metrics['waypoint_points'] / error_metrics['total_points'] * 100) if \
            error_metrics['total_points'] > 0 else 0
        print(f"  Waypoint Phase Data Points: {error_metrics['waypoint_points']} ({waypoint_percentage:.1f}% of total)")
    else:
        print("  No waypoint phase data available")

    # Transit phase statistics
    print("\n🚁 Transit Phase Analysis (When Drone is Moving Between Waypoints):")
    if error_metrics['transit_phase']['average'] is not None:
        print(f"  Average Path Deviation (3D): {error_metrics['transit_phase']['average']:.4f} m")
        print(f"  Average Horizontal Deviation (XY): {error_metrics['xy_plane']['transit']['average']:.4f} m")
        print(f"  Average Height Deviation (Z): {error_metrics['height']['transit']['average']:.4f} m")

        transit_percentage = (error_metrics['transit_points'] / error_metrics['total_points'] * 100) if \
            error_metrics['total_points'] > 0 else 0
        print(f"  Transit Phase Data Points: {error_metrics['transit_points']} ({transit_percentage:.1f}% of total)")
    else:
        print("  No transit phase data available")

    # Data quality information
    print("\n📊 Data Quality:")
    print(f"  Total Data Points: {error_metrics['total_points']}")
    print(f"  Outliers Removed: {error_metrics['outliers_removed']['overall']} overall, "
          f"{error_metrics['outliers_removed']['waypoint']} waypoint, "
          f"{error_metrics['outliers_removed']['transit']} transit")

    print("\n==== Error Analysis Complete ====\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='UAV RF Network Impact Test')

    parser.add_argument('--uri', type=str, default=None,
                        help='Crazyflie URI (if None, use default from environment)')
    parser.add_argument('--height', type=float, default=0.6,
                        help='Default flight height (meters)')
    parser.add_argument('--bandwidth', type=float, default=None,
                        help='Bandwidth limitation in Kbps')
    parser.add_argument('--latency', type=float, default=None,
                        help='Latency in milliseconds')
    parser.add_argument('--packet_loss', type=float, default=None,
                        help='Packet loss rate in percentage')
    parser.add_argument('--test', type=str, default='vertical',
                        choices=['square', 'vertical'],
                        help='Test type (square: square trajectory test, vertical: vertical ascent/descent test)')
    parser.add_argument('--side', type=float, default=1.0,
                        help='Side length of square trajectory (meters)')
    parser.add_argument('--disable-battery-check', action='store_true',
                        help='Disable the battery safety check (for testing only)')

    args = parser.parse_args()

    # Get network parameters directly from arguments
    bandwidth = args.bandwidth
    latency = args.latency
    packet_loss = args.packet_loss

    # Log parameter information
    logger.info(
        f"Parameters: URI={args.uri}, Bandwidth={bandwidth}Kbps, Latency={latency}ms, Packet Loss Rate={packet_loss}%")

    if args.disable_battery_check:
        logger.warning("Battery safety check disabled - USE WITH CAUTION")

    # Initialize Crazyflie drivers
    cflib.crtp.init_drivers()

    # Set Crazyflie URI
    uri = args.uri if args.uri else uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E6')

    # Create RF network simulator
    rf_sim = RFNetworkSimulator(
        bandwidth_kbps=bandwidth,
        latency_ms=latency,
        packet_loss_rate=packet_loss
    )
    try:
        # Connect to Crazyflie
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            connect_msg = f"Connected to Crazyflie: {uri}"
            print(connect_msg)
            logger.info(connect_msg)

            # Check battery before executing any test
            if not args.disable_battery_check:
                voltage, is_safe = BatteryDiagnostics.check_battery_voltage(scf)
                if not is_safe:
                    abort_msg = f"TEST ABORTED: Battery voltage ({voltage:.2f}V) below minimum threshold ({BATTERY_MIN_VOLTAGE}V)"
                    print(abort_msg)
                    logger.error(abort_msg)
                    return
                battery_msg = f"Battery check passed: {voltage:.2f}V (minimum: {BATTERY_MIN_VOLTAGE}V)"
                print(battery_msg)
                logger.info(battery_msg)
            else:
                print("Battery safety check disabled")
                logger.warning("Battery safety check disabled - proceeding without verification")

            # Set initial position
            initial_x = 0.0
            initial_y = 0.0
            initial_z = 0.0
            initial_yaw = 90

            set_initial_position(scf, initial_x, initial_y, initial_z, initial_yaw)
            reset_estimator(scf)

            # Wait for positioning system initialization
            wait_for_position_estimator(scf)

            optimize_controller_for_height(scf)

            time.sleep(1.0)

            # Choose execution based on test type
            if args.test == 'vertical':
                logger.info(f"Vertical ascent/descent test: 0.3m → 0.6m → 0.9m → 0.6m → 0.3m")
                results = run_vertical_sequence_with_rf_simulation(scf, rf_sim,
                                                                   base_x=initial_x,
                                                                   base_y=initial_y,
                                                                   base_z=initial_z,
                                                                   yaw=initial_yaw)
            elif args.test == 'square':
                # Run square trajectory test
                logger.info(f"Square trajectory test: Side length={args.side}m, Height={args.height}m")
                results = run_square_trajectory_with_rf_simulation(scf, rf_sim,
                                                                   side_length=args.side,
                                                                   height=args.height,
                                                                   base_x=initial_x,
                                                                   base_y=initial_y,
                                                                   base_z=initial_z,
                                                                   yaw=initial_yaw)

            # Output test results
            if isinstance(results, dict) and 'error' in results and results['error'] == 'low_battery':
                print("\nTest Results: ABORTED due to low battery")
                print(f"Battery Voltage: {results['battery_voltage']:.2f}V (Required: {results['battery_required']}V)")
                return

            print("\nTest Results Summary:")
            print(f"RF Conditions: Bandwidth={bandwidth}Kbps, Latency={latency}ms, Packet Loss Rate={packet_loss}%")
            print(f"Test Type: {args.test}")

            if 'battery_voltage' in results:
                print(f"Battery Voltage: {results['battery_voltage']:.2f}V")

            if args.test == 'square':
                print(f"Square Test: Side Length: {args.side}m, Height: {args.height}m")
            elif args.test == 'vertical':
                print(f"Vertical Sequence: 0.3m → 0.6m → 0.9m → 0.6m → 0.3m")

            if 'response_time' in results and results['response_time'] is not None:
                print(f"Command Response Time: {results['response_time']:.3f} seconds")

            if 'command_stats' in results:
                stats = results['command_stats']
                print(f"Command Success Rate: {stats['sent'] / stats['total_attempts'] * 100:.1f}%")

            if 'position_accuracy' in results and results['position_accuracy']:
                accuracy = results['position_accuracy']
                if accuracy['average_error'] is not None:
                    print(f"Average Position Error: {accuracy['average_error']:.4f} m")
                    print(f"Maximum Position Error: {accuracy['max_error']:.4f} m")

                    if 'transit_phase' in accuracy and accuracy['transit_phase']['average'] is not None:
                        print(f"Transit Phase Avg Error: {accuracy['transit_phase']['average']:.4f} m")
                    if 'waypoint_phase' in accuracy and accuracy['waypoint_phase']['average'] is not None:
                        print(f"Waypoint Phase Avg Error: {accuracy['waypoint_phase']['average']:.4f} m")
                    if 'config' in accuracy:
                        print(f"Error Calculation Mode: {accuracy['config']['vertical_mode']}")

            if 'result_file' in results:
                print(f"Detailed results saved to: {results['result_file']}")

    except KeyboardInterrupt:
        print("Test interrupted by user")
        logger.warning("Test interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        logger.error(f"Error during test: {e}")

    finally:
        logger.info("=== UAV RF Network Impact Test Completed ===")


if __name__ == "__main__":
    main()
