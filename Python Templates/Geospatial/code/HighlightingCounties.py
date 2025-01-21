################################################################################
# PRELIMINARY & UNPROOFED
# Not PRIVLEDGED & Not CONFIDENTIAL
# 
#   
# Case: Personal Interest/Training/Templates									
# Author: Tirth Bhatt
# Version Number:
# Date Created: 						
# Last Updated: 						
#   
# Audited: No
# To run from cmd: python "C:\Users\tbhatt\Documents\Templates\Geospatial\code\HighlightingCounties.py"
################################################################################

# %% Imports and setup
import geopandas as gpd
import matplotlib.pyplot as plt
import string
import os
from shapely.geometry import box
import random

# %%
# Define a list of flat colors for highlighting
flat_colors = ['#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#34495e', '#16a085', '#27ae60', '#2980b9', '#8e44ad', '#2c3e50',
               '#f1c40f', '#e67e22', '#e74c3c', '#ecf0f1', '#95a5a6', '#f39c12', '#d35400', '#c0392b', '#bdc3c7', '#7f8c8d']

# %%
# Function Definition
def load_data_and_create_maps(letters_to_map='A'):
    # Set up file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Go up one level
    shapefile_path = os.path.join(parent_dir, 'input', 'tl_2024_us_county.shp')
    output_dir = os.path.join(parent_dir, 'output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the county shapefile
    if os.path.exists(shapefile_path):
        counties = gpd.read_file(shapefile_path)
        print(f"Successfully loaded shapefile from {shapefile_path}")
    else:
        print(f"Error: Shapefile not found at {shapefile_path}")
        print("Please ensure the file exists and the path is correct.")
        return

    # Filter out Alaska (02), Hawaii (15), and non-state territories
    states_to_include = [
        '01', '04', '05', '06', '08', '09', '10', '11', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23', '24',
        '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44',
        '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56'
    ]
    counties = counties[counties['STATEFP'].isin(states_to_include)]

    # Set the projection to Albers Equal Area
    counties = counties.to_crs('ESRI:102003')

    def create_map_for_letter(letter, save=True, show=False):
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Plot all counties in light gray
        counties.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.1)
        
        # Highlight counties starting with the given letter
        highlighted_counties = counties[counties['NAME'].str[0].str.lower() == letter.lower()]
        highlight_color = random.choice(flat_colors)
        highlighted_counties.plot(ax=ax, color=highlight_color, edgecolor='white', linewidth=0.1)
        
        # Set the extent to focus on the contiguous United States
        bbox = box(-2500000, -1500000, 2500000, 1500000)
        ax.set_xlim(bbox.bounds[0], bbox.bounds[2])
        ax.set_ylim(bbox.bounds[1], bbox.bounds[3])
        
        # Set title with improved formatting (even larger font size)
        ax.set_title(f"Counties Starting with '{letter.upper()}'", fontsize=36, fontweight='bold', fontname='Calibri')
        ax.axis('off')
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(output_dir, f"county_map_{letter.upper()}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved map to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    # Generate maps
    if letters_to_map == 'all':
        for letter in string.ascii_uppercase:
            create_map_for_letter(letter)
        print("All maps generated successfully!")
    elif isinstance(letters_to_map, str) and len(letters_to_map) == 1:
        create_map_for_letter(letters_to_map, save=True, show=True)
    else:
        print("Invalid input. Use a single letter or 'all' to generate maps for all letters.")

# %%
# Run the full code
load_data_and_create_maps('all')