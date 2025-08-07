#!/usr/bin/env python3
"""
Safe Compatible Diagnostic Bandwidth Test Tool
Tests bandwidth without causing drone takeoff
"""
import time
import json
import argparse
from datetime import datetime
import sys


def safe_bandwidth_test(cf_lib_path):
    """
    Diagnostic bandwidth test using the same approach as your flight code,
    but ensures the drone stays locked on the ground
    """
    try:
        sys.path.append(cf_lib_path)
        import cflib.crtp
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

        # Initialize
        cflib.crtp.init_drivers()

        # Configure test parameters
        uri = 'radio://0/80/2M/E7E7E7E7E6'
        test_duration = 10  # seconds

        # Create log for results
        results_log = []
        connection_start_time = time.time()

        print("Starting SAFE bandwidth test (drone will stay on ground)...")
        print(f"Will test multiple frequencies using your control method")
        print("-" * 50)

        # Connect to Crazyflie
        with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
            connection_time = time.time() - connection_start_time
            print(f"Connection time: {connection_time:.3f} seconds")

            # This prevents the drone from responding to movement commands
            print("Locking motors for safety (drone will NOT take off)")
            scf.cf.platform.send_arming_request(False)
            time.sleep(0.5)  # Allow time for the lock to take effect

            # Test different command rates
            test_frequencies = [10, 20, 50, 100, 200]  # Hz
            frequency_results = {}

            for freq in test_frequencies:
                interval = 1.0 / freq
                print(f"\nTesting frequency: {freq} Hz (sleep interval: {interval:.4f}s)")

                commands_sent = 0
                commands_succeeded = 0
                total_command_time = 0

                # because the drone is locked, but we can still measure bandwidth
                x, y, z = 0.0, 0.0, 0.0  # Setting z=0 for extra safety
                yaw = 0

                # Run test for this frequency
                test_start = time.time()
                test_end = test_start + 5.0  # 5 seconds per frequency

                while time.time() < test_end:
                    cmd_start = time.time()
                    try:
                        # Send position command
                        # This won't cause the drone to move due to the lock
                        scf.cf.commander.send_position_setpoint(x, y, z, yaw)
                        cmd_end = time.time()

                        commands_succeeded += 1
                        total_command_time += (cmd_end - cmd_start)

                        # Record detailed timing for the first few commands
                        if len(results_log) < 50:
                            results_log.append({
                                'frequency': freq,
                                'command_time_ms': (cmd_end - cmd_start) * 1000,
                                'elapsed': cmd_end - test_start
                            })
                    except Exception as e:
                        print(f"Command failed: {e}")

                    commands_sent += 1

                    # Calculate next command time
                    next_time = test_start + (commands_sent * interval)
                    current_time = time.time()

                    if next_time > current_time:
                        time.sleep(next_time - current_time)

                # Calculate results for this frequency
                test_actual_time = time.time() - test_start
                achieved_rate = commands_succeeded / test_actual_time
                success_rate = commands_succeeded / commands_sent * 100 if commands_sent > 0 else 0
                avg_command_time = (total_command_time / commands_succeeded) * 1000 if commands_succeeded > 0 else 0

                # Calculate bandwidth based on position command size
                command_size_bytes = 16
                achieved_bandwidth = (achieved_rate * command_size_bytes * 8) / 1000  # kbps

                # Store results
                frequency_results[freq] = {
                    'target_rate': freq,
                    'achieved_rate': achieved_rate,
                    'commands_sent': commands_sent,
                    'commands_succeeded': commands_succeeded,
                    'success_rate': success_rate,
                    'avg_command_time_ms': avg_command_time,
                    'bandwidth_kbps': achieved_bandwidth
                }

                print(f"Results for {freq} Hz:")
                print(f"- Achieved rate: {achieved_rate:.1f} Hz ({(achieved_rate / freq * 100):.1f}% of target)")
                print(f"- Success rate: {success_rate:.1f}%")
                print(f"- Avg command time: {avg_command_time:.2f} ms")
                print(f"- Estimated bandwidth: {achieved_bandwidth:.2f} kbps")

            # Send stop command and confirm drone is still locked
            print("\nTest complete. Ensuring drone remains locked...")
            scf.cf.commander.send_stop_setpoint()
            scf.cf.platform.send_arming_request(False)

            # Find optimal frequency
            optimal_freq = 10
            max_success_rate = 0

            for freq, result in frequency_results.items():
                # Look for highest frequency with at least 95% success rate
                if result['success_rate'] >= 95 and result['achieved_rate'] / freq >= 0.9:
                    if freq > optimal_freq:
                        optimal_freq = freq

                # Track max success rate for fallback
                if result['success_rate'] > max_success_rate:
                    max_success_rate = result['success_rate']

            # If no frequency met our criteria, pick the one with highest success
            if optimal_freq == 10 and max_success_rate < 95:
                for freq, result in frequency_results.items():
                    if result['success_rate'] == max_success_rate:
                        optimal_freq = freq
                        break

            # Get the overall max bandwidth achieved
            max_bandwidth = 0
            for result in frequency_results.values():
                if result['bandwidth_kbps'] > max_bandwidth:
                    max_bandwidth = result['bandwidth_kbps']

            return {
                'connection_time': connection_time,
                'frequency_results': frequency_results,
                'optimal_frequency': optimal_freq,
                'max_bandwidth_kbps': max_bandwidth,
                'command_logs': results_log
            }

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Safe Compatible Crazyflie Bandwidth Test')
    parser.add_argument('--cf-lib-path', type=str, default='/opt/crazyflie-lib-python',
                        help='Crazyflie library path')

    args = parser.parse_args()

    print("=== SAFE Crazyflie Bandwidth Test (Compatible with Your Flight Code) ===")
    print("NOTE: This tool will not cause the drone to fly. It stays locked on ground.")
    print()

    # Run the test
    test_results = safe_bandwidth_test(args.cf_lib_path)

    if test_results:
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"bandwidth_test_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(test_results, f, indent=2)

        print(f"\nResults saved to: {filename}")

        # Print recommendations
        print("\n=== Recommendations for Your Flight Code ===")

        optimal_freq = test_results['optimal_frequency']
        max_bandwidth = test_results['max_bandwidth_kbps']

        print(f"Optimal control frequency: {optimal_freq} Hz")
        print(f"Maximum achieved bandwidth: {max_bandwidth:.2f} kbps")

        if max_bandwidth < 5:
            print("\n! CRITICAL: Very low bandwidth detected")
            print("- Check for RF interference")
            print("- Try different radio channels")
            print("- Reduce control frequency to 10Hz maximum")
        elif max_bandwidth < 20:
            print("\n! Warning: Limited bandwidth detected")
            print(f"- Limit control frequency to {min(optimal_freq, 20)} Hz")
            print("- Consider improving RF conditions")
        else:
            print(f"\n+ Good bandwidth detected")
            print(f"- Recommended control frequency: {optimal_freq} Hz")

        # Specific code recommendation
        print("\nUpdate your control loop with:")
        print(f"```python")
        print(f"# Send command with direct target position")
        print(f"command_stats['total_attempts'] += 1")
        print(f"success = rf_sim.send_position_with_conditions(cf, x, y, z, yaw)")
        print(f"if success:")
        print(f"    command_stats['sent'] += 1")
        print(f"else:")
        print(f"    command_stats['dropped'] += 1")
        print(f"")
        print(f"# Sleep for optimal timing")
        print(f"time.sleep({1.0 / optimal_freq:.4f})  # {optimal_freq} Hz control frequency")
        print(f"```")

    else:
        print("Test failed. Check connections and try again.")


if __name__ == "__main__":
    main()
