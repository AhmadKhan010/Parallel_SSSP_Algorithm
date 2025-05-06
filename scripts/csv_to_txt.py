# /home/ahmed/Desktop/PDC/Parallel_SSSP_Algorithm/csv_to_txt.py
import csv
import sys
import os

def convert_csv_to_txt(csv_file_path, output_path=None):
    """
    Convert a CSV file to a TXT file with the first 3 columns only (source, destination, weight)
    and add the number of nodes and edges at the top of the file.
    """
    if output_path is None:
        # Create output filename by replacing .csv with .txt
        base_name = os.path.splitext(csv_file_path)[0]
        output_path = "{}.txt".format(base_name)
    
    edges = []
    nodes = set()
    
    # Read the CSV file
    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Skip header if it exists
        try:
            header = next(csv_reader)
            has_header = True
        except StopIteration:
            print("Warning: CSV file appears to be empty.")
            return
        
        # Process each row
        for row in csv_reader:
            if len(row) >= 3:  # Ensure we have at least the 3 required columns
                source, dest, weight = row[0], row[1], row[2]
                
                # Try to convert to integers
                try:
                    source = int(source)
                    dest = int(dest)
                    weight = float(weight)
                    
                    # Convert negative weights to positive
                    if weight < 0:
                        weight = abs(weight)
                        print("Warning: Converted negative weight {} to positive {} for edge ({}, {})".format(-weight, weight, source, dest))
                    
                    edges.append((source, dest, weight))
                    nodes.add(source)
                    nodes.add(dest)
                except ValueError:
                    print("Warning: Skipping row {} due to invalid values.".format(row))
    
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    # Write to TXT file
    with open(output_path, 'w') as txtfile:
        # Write header with number of nodes and edges
        txtfile.write("{} {}\n".format(num_nodes, num_edges))
        
        # Write edges
        for source, dest, weight in edges:
            txtfile.write("{} {} {}\n".format(source, dest, weight))
    
    print("Conversion completed. Identified {} nodes and {} edges.".format(num_nodes, num_edges))
    print("Output written to {}".format(output_path))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_to_txt.py <csv_file_path> [output_txt_path]")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(csv_file_path):
        print("Error: File '{}' not found.".format(csv_file_path))
        sys.exit(1)
    
    convert_csv_to_txt(csv_file_path, output_path)