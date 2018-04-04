from PIL import Image, ImageChops
from pylab import *
import math
import numpy as np
import glob
import cv2
from sklearn.externals import joblib

f1 = [[-1,-1,1,1],[-1,-1,1,1],[-1,-1,1,1],[-1,-1,1,1]]
f2 = [[1,1,1,1],[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1]]
f3 = [[-1,-1,1,1],[-1,-1,1,1],[1,1,-1,-1],[1,1,-1,-1]]
f4 = [[-1,1,1,-1],[-1,1,1,-1],[-1,1,1,-1],[-1,1,1,-1]]
f5 = [[-1,-1,-1,-1],[1,1,1,1],[1,1,1,1],[-1,-1,-1,-1]]
f6 = [[-1,1,1,-1],[-1,1,1,-1],[1,-1,-1,1],[1,-1,-1,1]]
f7 = [[1,1,-1,-1],[-1,-1,1,1],[-1,-1,1,1],[1,1,-1,-1]]
f8 = [[1,-1,-1,1],[-1,1,1,-1],[-1,1,1,-1],[1,-1,-1,1]]
f9 = [[1,-1,1,-1],[1,-1,1,-1],[1,-1,1,-1],[1,-1,1,-1]]
f10 = [[-1,-1,-1,-1],[1,1,1,1],[-1,-1,-1,-1],[1,1,1,1]]
f11 = [[1,-1,1,-1],[1,-1,1,-1],[-1,1,-1,1],[-1,1,-1,1]]
f12 = [[1,1,-1,-1],[-1,-1,1,1],[1,1,-1,-1],[-1,-1,1,1]]
f13 = [[-1,1,-1,1],[1,-1,1,-1],[1,-1,1,-1],[-1,1,-1,1]]
f14 = [[1,-1,-1,1],[-1,1,1,-1],[-1,1,1,-1],[-1,-1,1,1]]
f15 = [[-1,1,-1,1],[1,-1,1,-1],[-1,1,-1,1],[1,-1,1,-1]]

def grayScale(image):
    newImage = Image.new('L', image.size)
    for y in range(0, image.size[1]):
        for x in range(0, image.size[0]):
            r,g,b = image.getpixel((x,y))
            g = round((r+g+b)/3)
            newImage.putpixel((x,y),g)

    return newImage

def image_to_massive(source):
    sourc = source.load()
    width = source.size[0]  # Определяем ширину.
    height = source.size[1]  # Определяем высоту.
    massive = np.zeros((height, width))
    i=0
    for x in range(width):
        for y in range(height):
            gray = source.getpixel((x, y))
            massive[y][x] = gray
        #i = i+1
    return massive

def minimum(source):
    #minimum = 255
    minimum = source.min()
    #for x in range(source.shape[0]):
     #   for y in range(source.shape[1]):
      #      #gray = source.getpixel((x, y))
       #     if(source[x][y]<minimum):
        #        minimum = source[x][y]
    for x in range(source.shape[0]):
        for y in range(source.shape[1]):
            source[x][y] = source[x][y] - minimum
            #if(source[x][y]<0):
             #   source[x][y]=0
    return source

def maximum(source):
    #maximum = 0
    maximum = source.max()
    #for x in range(source.shape[0]):
     #   for y in range(source.shape[1]):
      #      #gray = source.getpixel((x, y))
       #     if(source[x][y]>maximum):
        #        maximum = source[x][y]
    for x in range(source.shape[0]):
        for y in range(source.shape[1]):
            source[x][y] = round(source[x][y]/maximum, 1)
    return source

def norm(image):
    massive = image_to_massive(image)
    #print(massive)
    massive = minimum(massive)
    massive = maximum(massive)
    # np.savetxt('test1.txt', massive, delimiter=' ')
    # np.set_printoptions(threshold=np.nan)
    #print(massive)
    return massive

def q_preobr(source):
    q = 4
    N = int(source.shape[1]/4)#stolb
    M = int(source.shape[0]/4)#stroka
    result = np.zeros((q,q))
    sum = []
    for i in range(q):#stolb
        for j in range(q):#stroka
            sum = 0
            for y in  range(i*M,(i+1)*M):
                for x in range(j*N,(j+1)*N):
                    sum = sum + source[y][x]
            result[i][j] = sum

    return result

def m_create(source, f):
    res = 0
    for x in range(source.shape[0]):
        for y in range(source.shape[1]):
            res = res + source[x][y]*f[x][y]
    return round(res, 1)

def m_mass_create(source):
    mass = []
    #mass.append(m_create(source, F0))
    mass.append(m_create(source, f1))
    mass.append(m_create(source, f2))
    mass.append(m_create(source, f3))
    mass.append(m_create(source, f4))
    mass.append(m_create(source, f5))
    mass.append(m_create(source, f6))
    mass.append(m_create(source, f7))
    mass.append(m_create(source, f8))
    mass.append(m_create(source, f9))
    mass.append(m_create(source, f10))
    mass.append(m_create(source, f11))
    mass.append(m_create(source, f12))
    mass.append(m_create(source, f13))
    mass.append(m_create(source, f14))
    mass.append(m_create(source, f15))
    return mass


def getKeyPoints(source_name):
    source = Image.open(source_name)
    grey = grayScale(source)
    height = grey.size[1]
    width = grey.size[0]
    matrix = norm(grey)
   # print(matrix)
    res = []
    deviations = []
    points = []
    u = []
    i=1
    k=1
    windowSize=16
    while i<(height-windowSize+1):
        j=1
        while j < (width - windowSize+1):
            windowMatrix = matrix[i:i + windowSize, j:j + windowSize]
          #  print("windowMatrix")
           # print(windowMatrix)
            u = m_mass_create(q_preobr(windowMatrix))
            #print("u")
            #print(u)
            points.append([j,i])
            deviations.append(np.std(u))

            #k = k + 1
            j = j + 4
        i=i+4
    maxDev = max(deviations)
   # print(maxDev)
    for x in range(0, len(deviations)):
        if deviations[x] > maxDev * 0.6:
            res.append(points[x])
            #print(points[x].x, points[x].y)
            source.putpixel(points[x], 255)
            source.putpixel((int(points[x][0]+windowSize/2-1), int(points[x][1]+windowSize/2-1)), 255)
            source.putpixel((int(points[x][0] + windowSize / 2), int(points[x][1] + windowSize / 2 - 1)), 255)
            source.putpixel((int(points[x][0] + windowSize / 2-1), int(points[x][1] + windowSize / 2 )), 255)
            source.putpixel((int(points[x][0] + windowSize / 2), int(points[x][1] + windowSize / 2)), 255)

    source.save('ggg.jpg')
   # imshow(source)
   # show()
   # print(res)
    return res

original = []
table = np.zeros((415, 20))

j=1

def compare(shot):
    global j
    global original
    global table
   # print(shot)
   # print(original)
    i=0
    table[:,0]=1
    for dot in original:
        if dot in shot:

            table[i][j]=1
        else:
            table[i][j] = 0
        i=i+1;

        #table[0][ji]=1
        #i = i + 1
       # if dot in shot:
        #    table[i][j]=1
         #   i=i+1
        #else:
         #  table[i][j] = 0
    j=j+1
   # print(table)

def test():
    imageFolderPath = 'outside'
    imagePath = glob.glob(imageFolderPath + '/*.jpeg')
    global original
    original = getKeyPoints('out.jpeg')
    print(len(original))
    for img in imagePath:
       compare(getKeyPoints(img))

def final():
    global table
    #print( table)
    keyp = []
    h, w = table.shape
    sum=0
    reswi = 0
    perc = 0
    true_count=0;
    false_count=0;
    sum = np.sum(table,axis=1)
  #  print(sum)
    for el in sum:
        if el>=50:
            keyp.append(True)
            true_count += 1;
        if el>0 and el<50:
            keyp.append(False)
            false_count += 1;
    print(true_count);
    print(false_count);
    print(true_count*100/(true_count+false_count))


      #  perc=sum*100/w
       # if(perc>=50):
        #    keyp.append(True)
         #   true_count+=1;
        #else:
         #   if perc > 0 and perc<50:
          #      keyp.append(False)
           #     false_count+=1;
        #sum=0
       # print(perc)
        #print(true_count)
  #  print(true_count*100/(true_count+false_count)+1)
      #      reswi += table[hi][wi]
       # perc = 100 * reswi / w
       # print(perc)
        #reswi = 0


test()
final()