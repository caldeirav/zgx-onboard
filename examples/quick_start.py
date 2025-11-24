"""
Quick start example for ZGX Onboard.

This script demonstrates basic usage of the package.
"""

from zgx_onboard import load_config, setup_logging
from zgx_onboard.utils.config import get_settings


def main():
    """Main function demonstrating basic package usage."""
    # Set up logging
    setup_logging(log_level="INFO", console_output=True)
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    print(f"Device: {config['hardware']['device']}")
    print(f"Batch size: {config['training']['batch_size']}")
    
    # Get settings from environment
    print("\nLoading settings from environment...")
    settings = get_settings()
    print(f"Device from env: {settings.device}")
    print(f"Batch size from env: {settings.batch_size}")
    
    print("\nQuick start example completed successfully!")
    print("You can now start building your AI experiments.")


if __name__ == "__main__":
    main()

