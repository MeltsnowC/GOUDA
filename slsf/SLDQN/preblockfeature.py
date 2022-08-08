# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:47:08 2021

@author: Chenghongyi
"""
import mydata
import pandas as pd
import numpy as np
def preblock_line_feature():
    Allfeature = mydata.AllBLOCK
    print(Allfeature)
    print(len(Allfeature))
    length = len(Allfeature)
    toaldata = np.full((100,127),0)

    
    
    print(toaldata)
    print(len(toaldata))
    print(toaldata[1][:])
    print(len(toaldata[1]))
    
    
    
    #with open('.\data\word_vector.csv') as wordfile:
      #   for line in  wordfile:
     #        print(line)
    
    
    
    data = pd.read_csv(r'.\data\word_vector.csv')   #打开一个csv，得到data对象
    title = data.columns
    titlelist = []
    print(title)#获取列索引值
    #创建标题列表
    for t in title:
        titlelist.append(t)
    titlelist = titlelist[1:]
    print(titlelist)
    li =0
    for block in Allfeature:
        #print(block)
        if block in titlelist:
            #print(block)
            col = data[block]
            print(col)
            for num in col.index:
                print('num: '+ str(num))
                print('col[num]: '+str(col[num]))
                toaldata[num][li] = col[num]
            print('----------------toaldata('+str(li)+')----------------------')
            print(toaldata[:][li])
        if li>100:
            break
        li = li+1
                #toaldata[li][:len(col)] = num
            #print(col)
          #  titlelist[li] = data[block]
    print(titlelist)
    #print(toaldata)
    for liste in toaldata:
        print(liste)
    df = pd.DataFrame(toaldata, columns=Allfeature)
    df.to_csv(r'.\data\finalfeature.csv')
  
#data1 = data['name']#获取name列的数据
#data['new'] = data1 #将数据插入新列new
#data.to_csv(r'.\data\word_vector.csv',mode = 'a',index =False)
#保存到csv,  mode=a，以追加模式写入,header表示列名，默认为true,index表示行名，默认为true，再次写入不需要行名
#print(data)




#f = pd.read_csv('.\data\word_vector.csv')
#print(f)
#row = f.head(0)
#print(row)
#for i in row:
 #   print(row)