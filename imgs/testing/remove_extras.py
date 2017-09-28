import sys
import os

d = sys.argv[1]
d = './' + str(d)
count = 0

for subdir, dirs, files in os.walk(d):
    for f in files:
        path = os.path.join(subdir, f)
        #print path
        n = f.split('.')[0]
        cmd = 'rm {}'.format(path)
        n = int(n)
        if n < 1000:
            os.system(cmd)
            continue
        count +=1
        if count > 20:
            os.system(cmd)
