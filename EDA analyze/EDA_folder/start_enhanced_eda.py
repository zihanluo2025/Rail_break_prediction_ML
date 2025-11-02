"""
Startup Script for Enhanced Railway EDA Tool
Provides a user-friendly interface to run the enhanced analysis
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print the tool banner"""
    print("=" * 80)
    print("ENHANCED RAILWAY EDA TOOL")
    print("Multi-Table Data Analysis for Real Railway Sensor Data")
    print("=" * 80)
    print()

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'plotly', 'scipy', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstalling missing packages...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("Error installing packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def run_enhanced_eda():
    """Run the enhanced EDA analysis"""
    print("Starting Enhanced Railway EDA Analysis...")
    print()
    
    try:
        # Import and run the enhanced EDA tool
        from enhanced_railway_eda import EnhancedRailwayEDA
        
        # Get data directory
        data_dir = input("Enter the path to your data directory (or press Enter for 'datas'): ").strip()
        if not data_dir:
            data_dir = "datas"
        
        # Get output directory
        output_dir = input("Enter output directory name (or press Enter for 'real_data_output'): ").strip()
        if not output_dir:
            output_dir = "real_data_output"
        
        print(f"\nUsing data directory: {data_dir}")
        print(f"Output will be saved to: {output_dir}")
        print()
        
        # Initialize and run analysis
        eda_tool = EnhancedRailwayEDA(data_dir, output_dir)
        eda_tool.run_complete_analysis()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required files are in the same directory.")
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()

def run_quick_test():
    """Run a quick test of the data loader"""
    print("Running Quick Data Loader Test...")
    print()
    
    try:
        from multi_table_loader import MultiTableRailwayDataLoader
        
        # Test with sample data directory
        test_dir = "datas"
        if not os.path.exists(test_dir):
            print(f"Test directory {test_dir} not found.")
            return
        
        print(f"Testing data loader with directory: {test_dir}")
        
        # Initialize loader
        loader = MultiTableRailwayDataLoader(test_dir)
        
        # Load tables
        tables = loader.load_all_tables()
        
        if tables:
            print(f"\nSuccessfully loaded {len(tables)} tables:")
            for table_name, df in tables.items():
                print(f"  - {table_name}: {len(df)} rows, {len(df.columns)} columns")
            
            # Show overview
            print("\n" + "="*50)
            print("DATA OVERVIEW")
            print("="*50)
            overview = loader.get_data_overview()
            print(overview)
            
        else:
            print("No tables were loaded.")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure multi_table_loader.py is in the same directory.")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

def show_data_info():
    """Show information about the data structure"""
    print("RAILWAY DATA STRUCTURE INFORMATION")
    print("=" * 50)
    print()
    
    print("Your railway data consists of 7 related tables:")
    print()
    
    data_info = {
        'wagondata.csv': {
            'description': 'Main sensor data with measurements',
            'key_fields': 'BaseCode, KMLocation, RecordingDateTime, Twist14m, BounceFrt, BounceRr, Speed, etc.',
            'purpose': 'Core sensor readings for analysis'
        },
        'railbreaklocations.csv': {
            'description': 'Rail break location mapping',
            'key_fields': 'BaseCode, SectionBreakStartKM, SectionBreakFinishKM',
            'purpose': 'Maps specific track sections'
        },
        'trainingcontext.csv': {
            'description': 'Training data context with targets',
            'key_fields': 'BaseCode, SectionBreakStartKM, target (0/1), rul, break_date',
            'purpose': 'Historical data for predictive models'
        },
        'testcontext.csv': {
            'description': 'Test data context',
            'key_fields': 'BaseCode, SectionBreakStartKM, RecordingDate',
            'purpose': 'Data for testing/evaluation'
        },
        'inferencecontext.csv': {
            'description': 'Inference data context',
            'key_fields': 'BaseCode, SectionBreakStartKM, target, rul',
            'purpose': 'Data for inference/prediction'
        },
        'basecodemap.csv': {
            'description': 'Base code mapping table',
            'key_fields': 'BaseCode, MappedBaseCode',
            'purpose': 'Maps BaseCode to numeric values'
        },
        'allrailbreaksmapped.csv': {
            'description': 'Comprehensive rail break mapping',
            'key_fields': 'BaseCode, location data',
            'purpose': 'Complete rail break information'
        }
    }
    
    for filename, info in data_info.items():
        print(f"📊 {filename}")
        print(f"   Description: {info['description']}")
        print(f"   Key Fields: {info['key_fields']}")
        print(f"   Purpose: {info['purpose']}")
        print()
    
    print("The Enhanced EDA Tool will:")
    print("1. Load and integrate all tables")
    print("2. Analyze sensor patterns and correlations")
    print("3. Identify rail break prediction features")
    print("4. Create track health visualizations")
    print("5. Generate comprehensive reports")
    print()

def show_menu():
    """Display the main menu"""
    while True:
        print("\n" + "=" * 50)
        print("ENHANCED RAILWAY EDA TOOL - MAIN MENU")
        print("=" * 50)
        print("1. Run Complete Enhanced EDA Analysis")
        print("2. Quick Data Loader Test")
        print("3. View Data Structure Information")
        print("4. Check Dependencies")
        print("5. Exit")
        print()
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            run_enhanced_eda()
        elif choice == '2':
            run_quick_test()
        elif choice == '3':
            show_data_info()
        elif choice == '4':
            if check_dependencies():
                print("All dependencies are satisfied!")
            else:
                print("Some dependencies are missing.")
        elif choice == '5':
            print("Thank you for using the Enhanced Railway EDA Tool!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
        
        if choice in ['1', '2']:
            input("\nPress Enter to continue...")

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies first
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return
    
    # Show menu
    show_menu()

if __name__ == "__main__":
    main()

