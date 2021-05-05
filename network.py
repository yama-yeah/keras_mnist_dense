from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input,Dropout,Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
#import matplotlib.pyplot as plt
from data import Data
import numpy as np
EPOCH=46
class Network():
  def __init__(self,load):
    if(load):
      self.load()
    else:
      self.model_create()
  def model_create(self):
    self.model = Sequential()
    self.model.add(Input(shape=(28,28)))
    self.model.add(Flatten())
    self.model.add(Dropout(0.2))
    self.model.add(Dense(128,activation="relu"))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(64,activation="relu"))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(32,activation="relu"))
    self.model.add(Dropout(0.5))
    self.model.add(Dense((10), activation="softmax"))
    self.model.compile(optimizer="rsmprop",loss="categorical_crossentropy")
  def fit(self,train,test,e):
    x,y=train
    v_x,v_y=test
    print(np.shape(y))
    self.model.fit(x=x,y=y,epochs=e,validation_data=(v_x,v_y))
  def mnist_fit(self):
    datas=Data()
    train,test=datas.mnist_data()
    self.fit(train,test,EPOCH)
  def draw(self):
    plot_model(self.model, to_file='model.png',show_shapes=True)
  def save(self):
    self.model.save("weight/unko_learning.hdf5")
  def load(self):
    self.model=load_model("weight/unko_learning.hdf5")
  def output(self,path):
    gray=load_img(path, color_mode="grayscale", target_size=(28,28),interpolation="bilinear")
    #gray.show()
    gray = ImageOps.invert(gray) # 値反転
    x=img_to_array(gray)
    x=np.reshape(x,(1,28,28))
    print(x)
    for i in range(28):
      for j in range(28):
        if(x[0][i][j]>0):
          print("■",end="")
        else:
          print("□",end="")
      print()
    #plt.imshow(gray,'gray')
    
    #print(x)
    
    y=self.model.predict(x)
    y=y[0]
    return np.argmax(y)+1
    
      


    


if __name__=="__main__":
  #begin first train
  net=Network(True)
  net.draw()
  net.mnist_fit()
  net.save()
  print(net.output("figure/sample2.png"))

