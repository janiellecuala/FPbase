#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 00:25:10 2025

@author: janiellecuala
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Embedded laser power measurements
LASER_POWER = {
    680: 4.0, 690: 6.5, 700: 6.6, 710: 6.3, 720: 6.3, 
    730: 7.0, 740: 8.5, 750: 11.0, 760: 13.2, 770: 13.7, 780: 13.5
}

def find_replicate_files(fluorophore_name):
    """Find data files for replicates based on folder structure"""
    found_files = []
    
    # Check if fluorophore directory exists
    fluorophore_dir = os.path.join(os.getcwd(), fluorophore_name)
    if not os.path.isdir(fluorophore_dir):
        print(f"Error: Directory for {fluorophore_name} not found at {fluorophore_dir}")
        return found_files
    
    # Look for subdirectories (replicates)
    subdirs = []
    for item in os.listdir(fluorophore_dir):
        subdir_path = os.path.join(fluorophore_dir, item)
        if os.path.isdir(subdir_path):
            subdirs.append((item, subdir_path))
    
    subdirs.sort()  # Sort to ensure consistent processing order
    
    # Look for result files in each replicate subdirectory
    for subdir_name, subdir_path in subdirs:
        # Expected file name would be subdir_name + "_Results.csv"
        expected_file = f"{subdir_name}_Results.csv"
        file_path = os.path.join(subdir_path, expected_file)
        
        if os.path.isfile(file_path):
            found_files.append((subdir_name, file_path))
            print(f"Found file: {file_path}")
        else:
            # Try alternative naming patterns
            for file in os.listdir(subdir_path):
                if file.endswith('.csv') and subdir_name.lower() in file.lower():
                    file_path = os.path.join(subdir_path, file)
                    found_files.append((subdir_name, file_path))
                    print(f"Found file: {file_path}")
                    break
    
    return found_files

def auto_detect_columns(headers):
    """Automatically assign columns to compartments based on keywords"""
    compartment_columns = {
        'transfected': [],
        'untransfected': [],
        'cytoplasm': []
    }
    
    for header in headers:
        header_lower = header.lower()
        
        # Check for untransfected nuclei first
        if 'untransfected' in header_lower and ('nuc' in header_lower):
            compartment_columns['untransfected'].append(header)
        # Check for transfected nuclei second
        elif ('transfected' in header_lower or 'nuc' in header_lower) and 'untransfected' not in header_lower:
            compartment_columns['transfected'].append(header)
        # Check for cytoplasm last
        elif 'cyto' in header_lower:
            compartment_columns['cytoplasm'].append(header)
    
    return compartment_columns

def process_fluorophore(fluorophore_name, use_log_scale=True):
    """Process all replicates for a fluorophore"""
    # Find all replicate files
    replicate_files = find_replicate_files(fluorophore_name)
    
    if not replicate_files:
        print(f"No data files found for {fluorophore_name}.")
        return None
    
    print(f"Found {len(replicate_files)} replicate(s) for {fluorophore_name}.")
    
    # Process each replicate file
    all_processed_data = []
    
    # Get wavelength configuration once for all replicates
    start_wavelength = int(input("\nEnter starting wavelength (default: 680): ") or "680")
    wavelength_step = int(input("Enter wavelength increment (default: 10): ") or "10")
    
    # Process each replicate
    for i, (replicate_name, file_path) in enumerate(replicate_files):
        try:
            print(f"\nProcessing replicate {replicate_name} from {file_path}")
            
            # Read the data
            data = pd.read_csv(file_path)
            headers = data.columns.tolist()
            
            # Auto-detect columns
            compartment_columns = auto_detect_columns(headers)
            
            # Show detected columns
            print("\nAutomatically detected columns for each compartment:")
            for compartment, columns in compartment_columns.items():
                print(f"  {compartment}: {', '.join(columns) if columns else 'None'}")
            
            # Process data points for this replicate
            replicate_data = []
            
            for index, row in data.iterrows():
                # Calculate wavelength
                wavelength = start_wavelength + (index * wavelength_step)
                
                # Skip wavelengths outside our range
                if wavelength not in LASER_POWER:
                    continue
                
                # Create result dictionary
                result = {'wavelength': wavelength, 'replicate': replicate_name}
                
                # Process each compartment
                for compartment, columns in compartment_columns.items():
                    # Extract values
                    values = [row[col] for col in columns if col in row and pd.notna(row[col])]
                    
                    if not values:
                        result[f'{compartment}_mean'] = 0
                        result[f'{compartment}_error'] = 0
                        continue
                    
                    # Calculate mean
                    mean = sum(values) / len(values)
                    
                    # Calculate standard error
                    std_error = 0
                    if len(values) > 1:
                        variance = sum((val - mean) ** 2 for val in values) / (len(values) - 1)
                        std_error = np.sqrt(variance) / np.sqrt(len(values))
                    
                    # Normalize by laser power
                    laser_power = LASER_POWER.get(wavelength, 1)
                    normalized_mean = mean / laser_power
                    normalized_error = std_error / laser_power
                    
                    # Convert to log scale if requested
                    if use_log_scale and normalized_mean > 0:
                        log_mean = np.log10(normalized_mean)
                        log_error = normalized_error / (normalized_mean * np.log(10))
                        result[f'{compartment}_mean'] = log_mean
                        result[f'{compartment}_error'] = log_error
                    else:
                        # Use linear scale
                        result[f'{compartment}_mean'] = normalized_mean
                        result[f'{compartment}_error'] = normalized_error
                
                replicate_data.append(result)
            
            all_processed_data.extend(replicate_data)
            print(f"Successfully processed {len(replicate_data)} data points for replicate {replicate_name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Combine replicates
    if all_processed_data:
        combined_data = combine_replicates(all_processed_data, fluorophore_name)
        return combined_data
    else:
        print("No data was successfully processed.")
        return None

def combine_replicates(replicate_data, fluorophore_name):
    """Combine data from multiple replicates with error propagation"""
    print(f"\nCombining data from all replicates for {fluorophore_name}...")
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(replicate_data)
    
    if df.empty:
        return []
    
    # Get unique wavelengths and compartments
    wavelengths = sorted(df['wavelength'].unique())
    compartments = ['transfected', 'untransfected', 'cytoplasm']
    
    # Prepare combined dataset
    combined_data = []
    
    # Process each wavelength
    for wavelength in wavelengths:
        # Get data for this wavelength
        wavelength_data = df[df['wavelength'] == wavelength]
        
        # Create result dictionary
        result = {'wavelength': wavelength}
        
        # Process each compartment
        for compartment in compartments:
            # Get means and errors for this compartment across replicates
            means = wavelength_data[f'{compartment}_mean'].tolist()
            errors = wavelength_data[f'{compartment}_error'].tolist()
            
            # Remove zeros (missing data)
            valid_data = [(mean, error) for mean, error in zip(means, errors) if mean != 0]
            
            if not valid_data:
                result[f'{compartment}_mean'] = 0
                result[f'{compartment}_upper'] = 0
                result[f'{compartment}_lower'] = 0
                continue
            
            valid_means = [item[0] for item in valid_data]
            valid_errors = [item[1] for item in valid_data]
            
            # Calculate mean of means
            combined_mean = sum(valid_means) / len(valid_means)
            
            # Propagate errors properly
            # 1. Squared errors from technical replicates
            squared_tech_errors = sum(error ** 2 for error in valid_errors)
            
            # 2. Calculate standard error across biological replicates
            bio_variance = sum((mean - combined_mean) ** 2 for mean in valid_means)
            bio_error = np.sqrt(bio_variance) / np.sqrt(len(valid_means)) if len(valid_means) > 1 else 0
            
            # 3. Combine both sources of error
            combined_error = np.sqrt(squared_tech_errors + bio_error ** 2)
            
            # Store in result
            result[f'{compartment}_mean'] = combined_mean
            result[f'{compartment}_upper'] = combined_mean + combined_error
            result[f'{compartment}_lower'] = combined_mean - combined_error
        
        combined_data.append(result)
    
    print(f"Combined data: {len(combined_data)} data points.")
    return combined_data

def plot_data(data, fluorophore_name, use_log_scale=True, y_range=None):
    """Plot the processed data with consistent styling"""
    if not data:
        print("No data to plot.")
        return None
    
    # Colors for compartments
    compartment_colors = {
        'transfected': '#8884d8',  # purple
        'untransfected': '#82ca9d',  # green
        'cytoplasm': '#ff7300'  # orange
    }
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    
    # Plot each compartment
    compartments = ['transfected', 'untransfected', 'cytoplasm']
    compartment_names = {
        'transfected': 'Transfected Nuclei',
        'untransfected': 'Untransfected Nuclei',
        'cytoplasm': 'Cytoplasm'
    }
    
    for compartment in compartments:
        # Skip empty compartments
        if f'{compartment}_mean' not in df.columns or all(df[f'{compartment}_mean'] == 0):
            print(f"Skipping {compartment} - no data available")
            continue
        
        # Mean line
        ax.plot(
            df['wavelength'], 
            df[f'{compartment}_mean'],
            color=compartment_colors[compartment],
            marker='o',
            markersize=4,
            label=compartment_names[compartment]
        )
        
        # Error bands - always show them
        ax.fill_between(
            df['wavelength'],
            df[f'{compartment}_lower'],
            df[f'{compartment}_upper'],
            color=compartment_colors[compartment],
            alpha=0.2
        )
    
    # Set labels and title
    scale_type = "Log" if use_log_scale else "Linear"
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(f'{scale_type} Normalized Intensity')
    ax.set_title(f'{scale_type} Spectral Analysis of {fluorophore_name}')
    
    # Set axis limits
    ax.set_xlim(680, 780)
    
    # Set y-axis range if provided
    if y_range:
        ax.set_ylim(y_range[0], y_range[1])
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend()
    
    # Return figure
    return fig

def main():
    print("=== Fluorophore Spectral Analysis Tool ===")
    
    # Get the current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Change directory if needed
    new_dir = input("Enter path to data directory (or press Enter to use current): ")
    if new_dir.strip():
        try:
            os.chdir(new_dir)
            print(f"Working directory changed to: {os.getcwd()}")
        except Exception as e:
            print(f"Error changing directory: {e}")
    
    # List available fluorophore directories
    print("\nAvailable directories:")
    for item in sorted(os.listdir()):
        if os.path.isdir(item):
            print(f"  - {item}")
    
    # Get fluorophore name
    fluorophore = input("\nEnter fluorophore name: ")
    
    # Choose scale
    use_log_scale = input("Use logarithmic scale? (y/n): ").lower() == 'y'
    scale_type = "logarithmic" if use_log_scale else "linear"
    print(f"Using {scale_type} scale for intensity values")
    
    # Set consistent y-axis range
    use_consistent_y = input("Set consistent y-axis range? (y/n): ").lower() == 'y'
    y_range = None
    if use_consistent_y:
        min_y = float(input("Enter minimum y-axis value: "))
        max_y = float(input("Enter maximum y-axis value: "))
        y_range = (min_y, max_y)
    
    # Process the fluorophore
    processed_data = process_fluorophore(fluorophore, use_log_scale)
    
    if processed_data:
        # Always plot the data
        fig = plot_data(processed_data, fluorophore, use_log_scale, y_range)
        
        # Save the figure
        if input("Save figure? (y/n): ").lower() == 'y':
            scale_prefix = "log" if use_log_scale else "linear"
            filename = input(f"Enter filename (default: {fluorophore}_{scale_prefix}_spectral_analysis.png): ") or f"{fluorophore}_{scale_prefix}_spectral_analysis.png"
            
            dpi = int(input("Enter DPI (default: 300): ") or "300")
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved as {filename}")
        
        # Show the figure
        if input("Show figure? (y/n): ").lower() == 'y':
            plt.show()
        else:
            plt.close(fig)
        
        # Save processed data
        if input("Save processed data to CSV? (y/n): ").lower() == 'y':
            scale_prefix = "log" if use_log_scale else "linear"
            data_filename = input(f"Enter filename (default: {fluorophore}_{scale_prefix}_processed_data.csv): ") or f"{fluorophore}_{scale_prefix}_processed_data.csv"
            
            pd.DataFrame(processed_data).to_csv(data_filename, index=False)
            print(f"Data saved as {data_filename}")
        
        print(f"Processing complete for {fluorophore}!")
    else:
        print(f"Could not process data for {fluorophore}.")

if __name__ == "__main__":
    main()