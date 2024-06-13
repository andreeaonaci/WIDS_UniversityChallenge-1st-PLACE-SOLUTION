import csv

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        data.append(header)
        for row in reader:
            data.append(row)
    return data

def write_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def filter_data(data):
    filtered_data = [data[0]]  # Retain the header row
    filtered_data.extend([row for row in data[1:] if (3 < float(row[-1]) < 1300)])
    return filtered_data

def main(input_file, output_file):
    data = read_csv(input_file)
    filtered_data = filter_data(data)
    write_csv(output_file, filtered_data)

if __name__ == "__main__":
    main("/kaggle/input/widsdata/train.csv", "output.csv")
