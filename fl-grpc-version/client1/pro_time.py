
f = open('out.txt', 'r')

time = []
for line in f:
    if 'time used' in line:
        time.append(float(line.split()[2][0:-1]))

print(time)
