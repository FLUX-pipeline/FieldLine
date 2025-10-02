# FLutils.py

def load_pos_file(pos_path,np):
    with open(pos_path, "r") as f:
        lines = f.readlines()      
    num_points = int(lines[0].strip())  # First line = number of points
    points = []
    fiducials = {"NA": [], "LPA": [], "RPA": []}
    for line in lines[1:]:  # Skip first line
        parts = line.strip().split() 
        if len(parts) == 4:
            label, x, y, z = parts
            coords = np.array([float(x), float(y), float(z)])
            if label in fiducials:  # Store multiple fiducial points
                fiducials[label].append(coords)
            else:  
                points.append(coords)
    points = np.array(points)
    if np.max(points) > 1.0 or np.min(points) < -1.0:
        points /= 100.0      
    for key in fiducials:
        fiducials[key] = np.array(fiducials[key])  # Convert list to array first
        if np.max(fiducials[key]) > 1.0 or np.min(fiducials[key]) < -1.0:
            fiducials[key] /= 100.0    
    return points, fiducials


def compute_avg_fiducials(fiducials,np):
    avg_fiducials = {}
    for key, value in fiducials.items():
        if len(value) > 0:
            avg_fiducials[key] = np.mean(value, axis=0)
        else:
            avg_fiducials[key] = None  # Handle cases where no fiducial data is available
    return avg_fiducials


def als_to_ras(als_coords,np):
    als_coords = np.asarray(als_coords)
    return np.column_stack([-als_coords[:, 1], als_coords[:, 0], als_coords[:, 2]])

def als_to_ras_f(als_coords,np):
    als_coords = np.asarray(als_coords)
    als_coords = als_coords.reshape(-1, 3)  
    return np.column_stack([-als_coords[:, 1], als_coords[:, 0], als_coords[:, 2]])


def comp_avg_fiducials(fid_array,np):
    fid_array = np.asarray(fid_array)
    if fid_array.shape[0] % 2 != 0 or fid_array.shape[1] != 3:
        raise ValueError("Input must be of shape (6, 3) or (even number, 3)")
    avg_fiducials = (fid_array[::2] + fid_array[1::2]) / 2
    return avg_fiducials


def extract_fake_fiducials(points,np):
    if len(points) < 6:
        raise ValueError("Not enough head shape points to extract fake fiducials (need at least 6).")

    fake_fiducials = {
        "FNA": np.array([points[0], points[3]]),  # Fake Nasion points
        "FLPA": np.array([points[1], points[4]]),  # Fake LPA points
        "FRPA": np.array([points[2], points[5]]),  # Fake RPA points
    }
    avg_fake_fiducials = {
        key: np.mean(value, axis=0) if len(value) > 0 else None
        for key, value in fake_fiducials.items()
    }

    return fake_fiducials, avg_fake_fiducials



def level_and_average_dig_points(transformed_helmet_reference_points,helmet_dig_points,np):

    if helmet_dig_points.shape != (16, 3):
        raise ValueError("helmet_dig_points should have shape (16, 3)")
    labels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
    averaged_points = {}
    for i, label in enumerate(labels):
        averaged_points[label] = np.mean([transformed_helmet_reference_points[i], transformed_helmet_reference_points[i + 8]], axis=0)
    
    return averaged_points