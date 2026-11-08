
import json,os,time
from color_print import *

dir1 = "/home/jncsnlp4/SSD2/ZJJ/VRest-data/data_manager/mathvistra/ours2v2/FQA"

dir2 = "/home/jncsnlp4/SSD2/ZJJ/VRest-data/data_manager/mathvistra/ours2v3/FQA"

list1 = []

for i in os.listdir(dir1):
    list1.append(i.split(".")[0])


list2 = []

for i in os.listdir(dir2):
    list2.append(i.split(".")[0])


list3 = []
list4 = []
list5 = []
for i in list1:
    if i not in list2:
        list3.append(i)

for i in list2:
    if i not in list1:
        list4.append(i)

for i in list2:
    if i in list1:
        list5.append(i)
        
print(len(list3))
print(len(list4))
print(len(list5))
print(sorted(list3,key=lambda x:int(x.split("Q")[-1])))
print()
printblue(f"{sorted(list4,key=lambda x:int(x.split('Q')[-1]))}")
print()
printyellow(f"{sorted(list5,key=lambda x:int(x.split('Q')[-1]))}")





