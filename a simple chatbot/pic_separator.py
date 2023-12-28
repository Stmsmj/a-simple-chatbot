import os
import shutil

dir_path = os.path.dirname(os.path.realpath(__file__))
path_test=os.path.join(dir_path,'pic_loc\\test.txt')
path_train=os.path.join(dir_path,'pic_loc\\train.txt')
path_pic=os.path.join(dir_path,'Images')

test=list(open(path_test,'r'))
train=list(open(path_train,'r'))

test_pics=[]
train_pics=[]

for num in range(len(train)):
    train[num]=os.path.join(path_pic,train[num])

for num in range(len(test)):
    test[num]=os.path.join(path_pic,test[num])

classes=os.listdir(path_pic)

try:
    os.mkdir(os.path.join(dir_path,'test'))
    os.mkdir(os.path.join(dir_path,'train'))    
except:
    shutil.rmtree(os.path.join(dir_path,'test'))
    shutil.rmtree(os.path.join(dir_path,'train'))
    os.mkdir(os.path.join(dir_path,'test'))
    os.mkdir(os.path.join(dir_path,'train'))       


for class_ in classes:
    os.mkdir(os.path.join(os.path.join(dir_path,'test'),class_))
    os.mkdir(os.path.join(os.path.join(dir_path,'train'),class_))

for dir in train:
    splitted_dir=dir.split('Images')
    new_dir=splitted_dir[0]+'train'+splitted_dir[1]
    shutil.copyfile(dir[:-1],new_dir[:-1])

for dir in test:
    splitted_dir=dir.split('Images')
    new_dir=splitted_dir[0]+'test'+splitted_dir[1]
    shutil.copyfile(dir[:-1],new_dir[:-1])









