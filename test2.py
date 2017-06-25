class Path:
    def __init__(self, path):
        self.current_path = path
        self.path_list = path.split('/')

    def __repr__(self):
        return self.current_path


    def cd(self, new_path):
        # Convert path to a list
        path_list_candidate = new_path.split('/')
        path_list = []

        for p in path_list_candidate:
            if p == '..':
                del self.path_list[-1]
                path_list = self.path_list
            else:
                path_list.append(p)

        if len(path_list) > 1:
            if any([not str.isalpha(p) for p in path_list[1:]]):
                raise ValueError('Folders has to be alphabetic')

        if path_list[0] != '':
            raise ValueError('Root has to be /')

        return Path('/'.join(path_list))



path = Path('/a/b/c/d')
path = path.cd('../x')
print(path and path.current_path)
path = path.cd('../../john/smith')
print(path)