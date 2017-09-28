import idx2numpy

# Reading
ndarr = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')

f_read = open('t10k-images-idx3-ubyte', 'rb')
ndarr = idx2numpy.convert_from_file(f_read)

s = f_read.read()
ndarr = idx2numpy.convert_from_string(s)

# Writing 
idx2numpy.convert_to_file('myfile_copy.idx', ndarr)

f_write = open('myfile_copy2.idx', 'w')
idx2numpy.convert_to_file(f_write, ndarr)

s = convert_to_string(ndarr)
