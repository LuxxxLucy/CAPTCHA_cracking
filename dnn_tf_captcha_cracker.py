import tensorflow as tf
import os
import json
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from pprint import pprint as pr
import numpy as np
input_size_w=180
input_size_h=50
final_size_w=23
final_size_h=7
TOTAL_EPOCH_NUM=500
BATCH_SIZE=50

def cost_calcul(y,y_):
    # return sum([ tf.log(tf.slice(y[:],[i*62],[62]))*tf.slice(y_[:],[i*62],[62])  for i in range(6)] )
    return -tf.reduce_sum([ tf.log(y[:,i*62:i*62+62])* y_[:,i*62:i*62+62 ]   for i in range(6)] )

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

class DataSource:
    def __init__(self):
        data=self.load_data()
        total_num=len(data)
        self.test_set=np.array(data[:100])
        self.train_set=np.array(data[100:])
        self.test_num=len(self.test_set)
        self.train_num=len(self.train_set)
        
        print("test set length",self.test_num)
        print("train set length",self.train_num)
        self.count=0
    
    def test_set_data(self):
        return np.array([it[1] for it in self.test_set ])
    def test_set_labels(self):
        return np.array([ it[0] for it in self.test_set ])
    def number_of_train_sample(self):
        return self.train_num

    def load_data(self):
        with open("data/labels.json","r") as f:
            labels=json.load(f)
        label_dict={ label.split(r"/")[2].split(".")[0]: labels[label] for label in labels }
        image_dict=dict()
        for file_name in os.listdir(os.getcwd()+os.path.sep+"data/image/"):
            if(file_name.split(".")[1]!="png"):
                print(file_name.split("."))
                continue
            img_id=file_name.split(".")[0]
            with open(os.getcwd()+"/data/image/"+file_name,"rb") as f:
                image=Image.open(f)
                image.load()
                image=np.array(image)
                image=self.convert_to_gray(image)
            image_dict[img_id]=image
        return [(self.text_2_vec(label_dict[img_id]),image_dict[img_id].flatten()[:50*180]  ) for img_id in image_dict ]
    
    def convert_to_gray(self,img):
    	if len(img.shape) > 2:
    		# gray = np.mean(img, -1)
    		r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    		return gray
    	else:
    		return img
    def next_batch(self,batch_size):
        if(self.count+batch_size>self.train_num):
            pre=self.count
            self.count=0
            return [ it[1] for it in self.train_set[pre:]],[it[0] for it in self.train_set[pre:]]
        else:
            pre=self.count
            self.count+=batch_size
            return [ it[1] for it in self.train_set[pre:pre+batch_size]],[it[0] for it in self.train_set[pre:pre+batch_size]]
    
    def text_2_vec(self,symbol_list):
        def vec_mapping_table(s):
            result=np.zeros([62])
            result[s]=1
            return result
        return np.hstack([ vec_mapping_table(it) for it in symbol_list])

if __name__ == "__main__":
    print("loading data")
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # TOTAL_BATCH_NUM=1000
    # BATCH_SIZE=100

    test_data=tf.zeros([1,60*200])


    data = DataSource()

    x = tf.placeholder(tf.float32, [None, input_size_h*input_size_w])
    x_image = tf.reshape(x, [-1,input_size_w,input_size_h,1])

    W_conv1 = weight_variable([5, 5, 1, 48])
    b_conv1 = bias_variable([48])

    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='SAME')+ b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],strides=[1, 1, 1, 1], padding='SAME')

    W_conv3 = weight_variable([5,5,64,128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)


    W_fc1 = weight_variable([final_size_w * final_size_h * 128, 3072])
    b_fc1 = bias_variable([3072])

    h_pool3_flat = tf.reshape(h_pool3, [-1, final_size_w * final_size_h * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


    W_fc2 = weight_variable([3072, 372])
    b_fc2 = bias_variable([372])

    y=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)


    y_ = tf.placeholder("float", [None,372])



    # cross_entropy = tf.reduce_sum(cost_calcul(y,y_))
    cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y,labels=y_))
    # cross_entropy = -
    print(tf.reduce_sum(y_*tf.log(y)))
    print(cost_calcul(y,y_))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.initialize_all_variables()

    print("now start building the graph")
    sess = tf.Session()
    sess.run(init)
    print("session initial ok")


    # summary_writer = tf.train.SummaryWriter('/tmp/graph_logs', sess.graph)


    for epoch in range(TOTAL_EPOCH_NUM):
        for batch in range(int(data.number_of_train_sample() / BATCH_SIZE)):

            batch_xs,batch_ys = data.next_batch(BATCH_SIZE)
            try:
                sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys} )
            except:
                print("well unknown error happened")
         # if epoch%5 == 0:
            with sess.as_default():
                eval_acc = accuracy.eval(feed_dict={x:batch_xs,y_:batch_ys})
                print("batch completed at",batch,"of",int(data.number_of_train_sample()/ BATCH_SIZE),"training eval accuracy",eval_acc)
         #            summary_str = session.run(merged_summary_op)
         #            summary_writer.add_summary(summary_str, total_step)


    print("training complete!!!")
    print (sess.run(accuracy, feed_dict={x: data.test_set_data(), y_: data.test_set_labels}))
