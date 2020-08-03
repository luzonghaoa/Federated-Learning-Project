
f=open('top_out.txt', 'r')


cpu=[]
mem=[]
flag = 0
for line in f:
    if flag == 1:
        if line != '\n':
            cpu.append(line.split()[8])
            mem.append(line.split()[9])
        else:
            cpu.append(0.0)
            mem.append(0.0)
        flag=0
    if 'PID' in line:
        flag = 1

out = open('pro_out.txt', 'w')
out.write(str(cpu))
out.write('\n')
out.write(str(mem))

out.close()
f.close()
