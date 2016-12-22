import numpy


inputFile = open('../data/simulationData_0.txt', 'r')
outputFile = open('../data/sensor_mean_std_max.txt', 'w')

touch_values = []
for i in range(0,48):
    touch_values.append([])
count = 0
for line in inputFile.readlines():
    touch_values_str = line.split('*')[-2].split('|')[-1].split(' ')
    if len(touch_values_str) == 48:
        for i in range(0,48):
            touch_values[i].append(float(touch_values_str[i]))
    else:
        print "Error"
    count = count+1
    #if count==2:
    #    break

for i in range(0,48):
    mean = numpy.mean(touch_values[i])
    std_dev = numpy.std(touch_values[i])
    amax = numpy.amax(touch_values[i])
    outputFile.write(repr(mean) + ' ' + repr(std_dev) +' ' + repr(amax) +  '\n')


inputFile.close()
outputFile.close()
