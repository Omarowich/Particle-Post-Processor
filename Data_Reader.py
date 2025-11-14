import pandas as pd


def read_particle_data(file_path, sheet_name=0):
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.dropna(subset=['x', 'y'])

    # Read columns named 'x' and 'y' for coordinates
    x_coords = df['x'].tolist()
    y_coords = df['y'].tolist()

    # Double list of coordinates: [[x1, y1], [x2, y2], ...]
    coordinates = list(zip(x_coords, y_coords))

    return coordinates

#file_path = 'path/to/your/excel_file.xlsx'
#coordinates = read_particle_data(file_path)
#print(coordinates)