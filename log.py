import os
import traceback


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, file, resume=False):
        self.file = None
        self.resume = resume
        try:
            if not os.path.exists(fpath):
                os.mkdir(fpath)
                print('Create a new directory.')
            else:
                print('This directory exists.')
        except BaseException as msg:
            print('Fail to creat the new directory' + msg)

        file = fpath + '/' + file

        if os.path.isfile(file):
            if resume:
                self.file = open(file, 'a')
            else:
                self.file = open(file, 'w')
        else:
            self.file = open(file, 'w')

    def append(self, target_str):
        if not isinstance(target_str, str):
            try:
                target_str = str(target_str)
            except:
                traceback.print_exc()
            else:
                print(target_str)
                self.file.write(target_str + '\n')
                self.file.flush()
        else:
            print(target_str)
            self.file.write(target_str + '\n')
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
