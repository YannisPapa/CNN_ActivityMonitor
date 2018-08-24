
import tensorflow as tf
from tensorflow import graph_util
import CNN_ActivityRecognition

def freeze():
    saver = tf.train.import_meta_graph('checkpoints-cnn/har.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess = tf.Session()
    saver.restore(sess, "checkpoints-cnn/har")
    
    output_node_names="y_pred"
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names.split(",")  
    )
    
    output_graph="checkpoints-cnn/har-model.pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    sess.close()
    
if __name__ == '__main__':
    CNN_ActivityRecognition.main()
    freeze()
    
    g = tf.GraphDef()
    g.ParseFromString(open("checkpoints-cnn/har-model.pb", "rb").read())
    #print(g)