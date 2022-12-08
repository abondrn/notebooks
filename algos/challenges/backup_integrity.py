import sys

class Directory:
    
    def __init__(self):
        self.num_files = 0
        self.subnodes = dict()
    
    def add_file(self, name):
        self.num_files += 1
    
    def get_folder(self, name):
        if name not in self.subnodes:
            self.subnodes[name] = Directory()
        return self.subnodes[name]
    
    def __repr__(self):
        return str(self.num_files)+str(self.subnodes)

def populate_fs():
    num_files = int(sys.stdin.readline())
    root = Directory()
    for _ in range(num_files):
        path = sys.stdin.readline().split('/')
        dir = root
        for i in range(len(path)-1):
            dir = dir.get_folder(path[i])
        dir.add_file(path[-1])
    return root
        
def matches(dir1, dir2):
    if dir1.num_files != dir2.num_files or len(dir1.subnodes) != len(dir2.subnodes):
        return False
    subdirs2 = list(dir2.subnodes.values())
    for subdir1 in dir1.subnodes.values():
        for i, subdir2 in enumerate(subdirs2):
            if matches(subdir1, subdir2):
                del subdirs2[i]
                break
        else:
            return False
    return True

original_fs = populate_fs()
encrypted_fs = populate_fs()
if matches(original_fs, encrypted_fs) and matches(original_fs, encrypted_fs):
    print('OK')
else:
    print('INVALID')