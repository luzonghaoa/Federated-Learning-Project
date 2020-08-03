
f = open('out.txt', 'r')

time = []
for line in f:
    if 'total_time_used' in line:
        print(line)

