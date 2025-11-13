#!/usr/bin/env python3
import sys

def allocate_until_crash():
    """Allocate memory until system crashes or throws an error."""
    allocated = []
    i = 0
    chunk_size = 10*1024 * 1024 * 1024  # 1GB chunks
    
    try:
        while True:
            # Allocate and touch memory to force real allocation
            chunk = bytearray(chunk_size)
            for j in range(0, len(chunk), 4096):
                chunk[j] = 0xFF
            allocated.append(chunk)
            i += 1
            print(f"Allocated {i * 10}GB", flush=True)
    except Exception as e:
        print(f"Failed after {i * 10}GB: {e}")
        return

if __name__ == "__main__":
    allocate_until_crash()