import csv

# Read the tab-separated file
with open('SoilQualityDataset.tab', 'r', encoding='utf-8') as infile:
    content = infile.read()

# Find where the actual data starts (after the data description)
data_start = content.find('Event\tLab label\tLatitude')
if data_start == -1:
    print("Could not find the data header. Please check the file format.")
    exit(1)

# Extract the data portion (header and rows)
data_content = content[data_start:]

# Split into lines
lines = data_content.strip().split('\n')

# Open a new CSV file for writing
with open('SoilQualityDataset.csv', 'w', newline='', encoding='utf-8') as outfile:
    csv_writer = csv.writer(outfile)
    
    # Process each line
    for line in lines:
        # Split by tabs and write as CSV
        values = line.split('\t')
        csv_writer.writerow(values)

print(f"Conversion complete. CSV file saved as 'SoilQualityDataset.csv'") 