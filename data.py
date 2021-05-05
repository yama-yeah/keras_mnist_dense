from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class Data():
  def __init__(self):
    (self.train_images, self.train_labels), (self.test_images, self.test_labels) = mnist.load_data()
  def mnist_data(self):
    self.train_labels = to_categorical(self.train_labels,10)
    self.test_labels = to_categorical(self.test_labels,10)
    return (self.train_images, self.train_labels), (self.test_images, self.test_labels)
  def draw_sample(self):
    sample,_=self.mnist_data()
    sample,l=sample
    for i in range(28):
      for j in range(28):
        print(sample[0][i][j],end=" ")
      print()
    print(l[0])
if __name__=="__main__":
  data=Data()
  data.draw_sample()