import os

file_path = 'test/left'
if os.path.isfile(file_path):
    print("File exists and can be accessed.")
else:
    print("File does not exist or cannot be accessed.")