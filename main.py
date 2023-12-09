import re
import os

directory = os.path.join('Database_arrows', 'train', 'right')

for filename in os.listdir(directory):
    new_filename = re.sub(r'left', r'right', filename)
    print(filename)
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
