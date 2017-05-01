import cv2
import os

files = os.listdir('sample-data/IMG')
files = [f for f in files if f.startswith('center')]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output.avi',fourcc, 20.0, (640,480))

count = 0
total = len(files)
for f in files:
    filename = os.path.join('sample-data/IMG/', f)
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    out.write(image)
    print('{} of {}'.format(count, total), end='\r')
    count += 1
    
out.release()
cv2.destroyAllWindows()


print('Done')