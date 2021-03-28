import csv
f = open("imu_data.txt", "r")
x_data = []
y_data = []
for x in f:
    if 'x' in x:
        x_data.append(float(x[4:]))
    if ' y:' in x: 
        y_data.append(float(x[4:]))

dict_data = {'x':x_data,'y':y_data}
csv_columns = ['x','y']
csv_file = "imu_clean.csv"
try:
    with open(csv_file, 'w') as f:
        w = csv.DictWriter(f, dict_data.keys())
        w.writeheader()
        w.writerow(dict_data)
except IOError:
    print("I/O error")

