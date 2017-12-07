import os, sys
import subprocess

nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

target = sys.argv[1]

target_dir = os.path.join('imgs', target)
correct = 0
incorrect = 0
total = 0
# dict that stores accuracy for each digit
accuracy_dict = dict()

for n in nums:
	num_right = 0
	num_total = 0
	curr_dir = os.path.join(target_dir, str(n))
	for subdir, dirs, files in os.walk(curr_dir):
		for f in files:
			img_path = os.path.join(subdir, f)
			#print img_path
			cmd = 'python guess_digit.py {}'.format(img_path)
			print(cmd)
			result = subprocess.run(cmd, stdout=subprocess.PIPE)
			result = result.stdout.decode('utf-8')
			result = result.rstrip()
			print(result)
			try:
				digit = int(result)
			except:
				print("Guess failed")
				continue
			if digit == n:
				correct += 1
				num_right += 1
			else:
				incorrect += 1
			total += 1
			num_total += 1
			accuracy = float(correct)/total * 100
			print('Accuracy: {0:.2f}%'.format(accuracy))
	accuracy_dict[n] = float(num_right)/num_total * 100
	
print('End Result:')
accuracy = float(correct)/total * 100
print('Accuracy: {0:.2f}%'.format(accuracy))

# display accuracy for each digit
for num in accuracy_dict:
	print('{0}: {1:.2f}%'.format(str(num), accuracy_dict[num]))