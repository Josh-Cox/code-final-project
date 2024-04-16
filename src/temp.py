import os
import re

path = '../models/ebm'

def main():
    biggest = 0.0
    best = ''
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(subdir, filename)
            
            if filename == 'results.txt':
                content = open(filepath).readlines()

                x = re.search(r'\.\d+', content[0])
                number = int(x.group()[1:])
                if number > biggest:
                    biggest = number
                    best = filepath
                print(filepath)
                print(number)
    print(best)
    print(biggest)
                    
            
if __name__ == '__main__':
    main()