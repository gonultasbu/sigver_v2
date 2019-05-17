from PIL import Image
import os
import tqdm

def find_largest(directory):
    sizes=[0,0]
    for root, dirs, filenames in (os.walk(directory, topdown=False)):
        for file in filenames:
            if (file.endswith(".jpg")):
                sizes_temp = [Image.open(os.path.join(root,file), 'r').size[0],Image.open(os.path.join(root,file), 'r').size[1]]
                if (sizes_temp[0]>sizes[0]):
                    sizes[0]=sizes_temp[0]
                else:
                    pass
                if (sizes_temp[1]>sizes[1]):
                    sizes[1]=sizes_temp[1]
                else:
                    pass
    print ("Max size within dataset is" + str(sizes[::-1]))
    return sizes[::-1]

if (__name__ == "__main__"):
    find_largest("C:\\Users\\Mert\\Documents\\GitHub\\sigver_bmg\\data\\GPDSSyntheticSignatures4k")