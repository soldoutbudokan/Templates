import re
import os

def organize_m3u(input_file, output_file):
    """
    Reorganize an M3U file by adding group tags based on channel name prefixes.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Ensure the file starts with #EXTM3U
    if not lines[0].strip() == '#EXTM3U':
        print("Error: Input file does not appear to be a valid M3U file.")
        return

    # Create output with the header
    output_lines = ['#EXTM3U\n']
    
    # Process lines
    i = 1  # Skip the header
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this is an EXTINF line
        if line.startswith('#EXTINF:'):
            # Extract channel name
            match = re.search(r'#EXTINF:-1,(.*)', line)
            if match and i + 1 < len(lines):
                channel_name = match.group(1).strip()
                url = lines[i + 1].strip()
                
                # Determine group based on prefix
                group = "Uncategorized"
                
                if channel_name.startswith('MM:'):
                    group = "Main Media"
                elif channel_name.startswith('WMM:'):
                    group = "World Main Media"
                elif channel_name.startswith('EVENTS'):
                    group = "Live Events"
                elif ':' in channel_name:
                    # Try to use the part before the colon as group
                    potential_group = channel_name.split(':', 1)[0].strip()
                    if potential_group and len(potential_group) < 20:  # Reasonable length for a group name
                        group = potential_group
                
                # Add the modified EXTINF line with group tag
                output_lines.append(f'#EXTINF:-1 group-title="{group}",{channel_name}\n')
                output_lines.append(f'{url}\n')
                
                i += 2  # Skip the URL line
            else:
                output_lines.append(line + '\n')
                i += 1
        else:
            output_lines.append(line + '\n')
            i += 1

    # Write the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"Organized M3U file saved to '{output_file}'")

if __name__ == "__main__":
    input_file = "tv.m3u"  # Your actual file name
    output_file = "tv2.m3u"
    organize_m3u(input_file, output_file)