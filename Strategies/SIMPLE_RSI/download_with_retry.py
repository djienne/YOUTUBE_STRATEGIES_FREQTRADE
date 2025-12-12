import subprocess
import re
import time
import sys
from datetime import datetime

def extract_ban_timestamp(error_message):
    """Extract the ban timestamp from the error message (in milliseconds)"""
    match = re.search(r'banned until (\d+)', error_message)
    if match:
        # Timestamp is in milliseconds, convert to seconds
        return int(match.group(1)) / 1000
    return None

def run_download_command():
    """Run the freqtrade download-data command with automatic retry on ban"""
    command = [
        "freqtrade",
        "download-data",
        "--timeframe", "1d",
        "--timerange", "20170817-",
        "--config", "./user_data/config_backtest.json"
    ]

    attempt = 1

    while True:
        print(f"\n{'='*70}")
        print(f"Attempt #{attempt}")
        print(f"Running: {' '.join(command)}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        ban_detected = False
        ban_timestamp = None

        try:
            # Run the command with real-time output monitoring
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Read character by character to detect ban ASAP
            output_buffer = ""

            while True:
                # Read one character at a time
                char = process.stdout.read(1)

                if not char:
                    # EOF - process finished
                    break

                # Print character immediately
                sys.stdout.write(char)
                sys.stdout.flush()

                # Add to buffer
                output_buffer += char

                # Check last 500 chars for ban message (to avoid checking entire history)
                check_window = output_buffer[-500:] if len(output_buffer) > 500 else output_buffer

                # Look for ban pattern
                if "banned until" in check_window and ("-1003" in check_window or "418" in check_window):
                    ban_detected = True
                    ban_timestamp = extract_ban_timestamp(output_buffer)

                    print("\n\n" + "="*70, flush=True)
                    print("⚠️  BAN DETECTED - KILLING PROCESS IMMEDIATELY!", flush=True)
                    print("="*70 + "\n", flush=True)

                    # Kill process NOW
                    process.kill()
                    process.wait()
                    break

            # Wait for process if not killed
            if not ban_detected:
                process.wait()

            # If banned, calculate wait time and retry
            if ban_detected:
                if ban_timestamp:
                    # We have a timestamp - use it
                    current_time = time.time()
                    wait_time = ban_timestamp - current_time

                    # Add 2 minute safety margin
                    margin_seconds = 2 * 60
                    wait_time += margin_seconds

                    if wait_time > 0:
                        ban_datetime = datetime.fromtimestamp(ban_timestamp)
                        resume_datetime = datetime.fromtimestamp(ban_timestamp + margin_seconds)

                        print(f"\n{'='*70}")
                        print(f"⚠️  IP BANNED DETECTED")
                        print(f"Banned until: {ban_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"Will resume at: {resume_datetime.strftime('%Y-%m-%d %H:%M:%S')} (with 2min margin)")
                        print(f"Waiting for {wait_time/60:.1f} minutes...")
                        print(f"{'='*70}\n")

                        # Wait until ban expires
                        time.sleep(wait_time)

                        print(f"\n✓ Ban period expired. Retrying...\n")
                        attempt += 1
                        continue
                    else:
                        print("\nBan timestamp already passed. Retrying immediately...")
                        attempt += 1
                        continue
                else:
                    # No timestamp extracted - wait 15 minutes by default
                    default_wait = 15 * 60  # 15 minutes in seconds

                    print(f"\n{'='*70}")
                    print(f"⚠️  IP BANNED DETECTED (timestamp not found)")
                    print(f"Waiting 15 minutes as a safety precaution...")
                    print(f"Will resume at: {datetime.fromtimestamp(time.time() + default_wait).strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*70}\n")

                    time.sleep(default_wait)

                    print(f"\n✓ Wait period expired. Retrying...\n")
                    attempt += 1
                    continue

            # If process completed successfully
            if process.returncode == 0:
                print(f"\n{'='*70}")
                print("✓ Command completed successfully!")
                print(f"{'='*70}\n")
                break
            else:
                # Check if it's a different error (not a ban)
                if not ban_detected:
                    print(f"\n{'='*70}")
                    print(f"✗ Command failed with return code {process.returncode}")
                    print("This doesn't appear to be a ban error. Check the output above.")
                    print(f"{'='*70}\n")
                    break

        except KeyboardInterrupt:
            print("\n\nScript interrupted by user (Ctrl+C)")
            try:
                process.kill()
            except:
                pass
            break
        except Exception as e:
            print(f"\nError running command: {e}")
            break

if __name__ == "__main__":
    print("\nFreqtrade Download Data with Auto-Retry on Ban")
    print("This script will automatically wait and retry if IP gets banned\n")
    run_download_command()
