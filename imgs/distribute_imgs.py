import sys
import os

### all 20 images were in testing to start with.  Script redistributes images so that training gets 10, validation gets 5 and testing gets 5 ###

nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

test_dir = './testing'
train_dir = './training'
valid_dir = './validation'

for n in nums:
    curr_dir = os.path.join(test_dir, str(n))
    print curr_dir
    training = 0
    validation = 0
    for subdir, dirs, files in os.walk(curr_dir):
        for f in files:
            img_path = os.path.join(subdir, f)
            #print img_path
            if training < 10:
                target_dir = os.path.join(train_dir, str(n))
                cmd = 'mv {} {}'.format(img_path, target_dir)
                #print cmd
                os.system(cmd)
                training += 1
            elif validation < 5:
                target_dir = os.path.join(valid_dir, str(n))
                cmd = 'mv {} {}'.format(img_path, target_dir)
                #print cmd
                os.system(cmd)
                validation += 1
