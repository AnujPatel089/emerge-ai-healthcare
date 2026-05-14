"""
Compatibility wrapper for the modular EmergAI data generator.

Preferred command:
    python -m src.generate_emergency_data

This wrapper keeps older commands working:
    python generate_emergency_data.py
"""

from src.generate_emergency_data import main


if __name__ == "__main__":
    main()

