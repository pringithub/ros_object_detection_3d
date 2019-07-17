#!/usr/bin/env python
## Author: Rohit
## Date: July, 25, 2017
# Purpose: Ros node to detect objects using tensorflow

import os
import sys
import cv2
import numpy as np
import tf as rostransform
import tf.msg as rostransform_msg
import geometry_msgs.msg
import math
try:
    import tensorflow as tf
except ImportError:
    print("unable to import TensorFlow. Is it installed?")
    print("  sudo apt install python-pip")
    print("  sudo pip install tensorflow")
    sys.exit(1)

# ROS related imports
import rospy
from std_msgs.msg import String , Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
import message_filters

# Object detection module imports
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
   

DEBUG_MODE=1 # 1=verbos 0=not_verbose
#
PUBLISH_IN_CAMERA_FRAME=True
#
IMAGE_HEIGHT=480
IMAGE_WIDTH=640
#
rospy.set_param('zinverse_depth_scaling', {'y': 1.0, 'z': 1.0})


 
# SET FRACTION OF GPU YOU WANT TO USE HERE
GPU_FRACTION = 0.4

######### Set model here ############
MODEL_NAME =  'ssd_mobilenet_v1_coco_11_06_2017'
# By default models are stored in data/models/
MODEL_PATH = os.path.join(os.path.dirname(sys.path[0]),'data','models' , MODEL_NAME)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_PATH + '/frozen_inference_graph.pb'
######### Set the label map file here ###########
LABEL_NAME = 'mscoco_label_map.pbtxt'
# By default label maps are stored in data/labels/
PATH_TO_LABELS = os.path.join(os.path.dirname(sys.path[0]),'data','labels', LABEL_NAME)
######### Set the number of classes here #########
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print(categories)


# Setting the GPU options to use fraction of gpu that has been set
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION





# Detection

class Detector:

    def __init__(self):
  
        self.bridge = CvBridge()
        self.sess = tf.Session(graph=detection_graph,config=config)

        self.depth_image_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.color_image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        ts = message_filters.TimeSynchronizer([self.depth_image_sub, self.color_image_sub], 10)
        ts.registerCallback(self.processing_cb)

        self.tf_listener = rostransform.TransformListener()

        self.image_pub = rospy.Publisher("debug_image",Image, queue_size=1)
        self.object_pub = rospy.Publisher("objects", Detection2DArray, queue_size=1)
        #using rostransform instead of tf for namespace reasons              
        self.pub_tf = rospy.Publisher("/tf", rostransform_msg.tfMessage, queue_size=1)  
        self.test_pub = rospy.Publisher("new_debug_image",Image, queue_size=1)
        self.debug_pub = rospy.Publisher("depth_debug_image",Image, queue_size=1)

    def processing_cb(self, depth_img_data, color_img_data):

        print('---------')

        data = color_img_data # for consistency with prev source

        objArray = Detection2DArray()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            depth_img_data.encoding = "mono16" # need this to work
            depth_image = self.bridge.imgmsg_to_cv2(depth_img_data, "mono16")
            #print('cv_image shape: '+str(cv_image.shape))
            #print('depth_image shape: '+str(depth_image.shape))
        except CvBridgeError as e:
            print(e)


        image=cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = np.asarray(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        objects=vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)

        objArray.detections =[]
        objArray.header=data.header
        object_count=1


        for i in range(len(objects)):
            object_count+=1
            obj_struct = self.object_predict(objects[i],data.header,image_np,cv_image)
            objArray.detections.append(obj_struct)

	    print(objects[i])

            # get depth of object and write debug info on cv_image
            avg_depth, center, cv_image = self.get_object_depth(obj_struct, cv_image, depth_image, i)
            if avg_depth == 0:
                continue

            # calculate 3D tf and publish it
            obj_id = objects[i][0]
            obj_category = category_index[obj_id]['name']
            self.object_tf_publisher(center, avg_depth, obj_category, i)
            
    
        self.object_pub.publish(objArray)
        
 
        if DEBUG_MODE == 1: 
 
            # /debug_image - base tensorflow image detection
            img=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            image_out = Image()
            try:
                image_out = self.bridge.cv2_to_imgmsg(img,"bgr8")
            except CvBridgeError as e:
                print(e)
            image_out.header = data.header
            self.image_pub.publish(image_out)

            depth_image_out = depth_image # TODO: remove this debug image and all instances hereafter
            try:
                cv_image_out = self.bridge.cv2_to_imgmsg(cv_image,"bgr8")
                depth_image_out = self.bridge.cv2_to_imgmsg(depth_image_out,"mono16")
            except CvBridgeError as e:
                print(e) 
            self.test_pub.publish(cv_image_out)
            self.debug_pub.publish(depth_image_out)


    def object_tf_publisher(self, center, avg_depth, obj_category, i):

        # (col,row) for rgb image

        u = center[1] - IMAGE_HEIGHT/2
        v = center[0] - IMAGE_WIDTH/2
        # u/v = y/z, so need a sclaing factor
        y_scaling_factor = rospy.get_param('zinverse_depth_scaling/y') 
        z_scaling_factor = rospy.get_param('zinverse_depth_scaling/y') 
        y = y_scaling_factor * v 
        z = z_scaling_factor * u
        try: 
            x = math.sqrt(avg_depth**2 - y**2 - z**2)
        except:
            print('[ERROR] calculating x')
            return # was continue in that loop 
 
        if DEBUG_MODE == 1:
            print('Object: '+obj_category+'_'+str(i))
            print('Pixel coords: '+str(u)+','+str(v))  
            print('3D Coords   : '+str(x)+','+str(y)+','+str(z)) 

            
     	t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
	t.child_frame_id = obj_category+'_'#+str(i)
 	t.transform.rotation.x = 0.0 
	t.transform.rotation.y = 0.0
	t.transform.rotation.z = 0.0 
	t.transform.rotation.w = 1.0
	#
        if PUBLISH_IN_CAMERA_FRAME:
            # publish object in camera frame
	    t.header.frame_id = "camera_link"
	    t.transform.translation.x = x / 1000.0
	    t.transform.translation.y = -y / 1000.0 # tf/rviz/cartesian defined opposite than image 
	    t.transform.translation.z = -z / 1000.0 # tf/rviz/cartesian defined opposite than image 
	    tfm = rostransform.msg.tfMessage([t])
	    self.pub_tf.publish(tfm)
        else:
            # publish object in map frame
            try:
	        (trans_map_to_cam, rot) = self.tf_listener.lookupTransform( \
                                              '/map', '/camera_link', rospy.Time(0))
                trans_cam_to_obj = (x / 1000.0, -y / 1000.0, -z / 1000.0)
                trans = np.add(trans_map_to_cam, trans_cam_to_obj) 
	        #
                t.header.frame_id = "map"
	        t.child_frame_id = obj_category+'_'#+str(i)
	        t.transform.translation.x = trans[0] 
	        t.transform.translation.y = trans[1]
	        t.transform.translation.z = trans[2]
	        tfm = rostransform.msg.tfMessage([t])
	        self.pub_tf.publish(tfm)

            except (rostransform.LookupException, rostransform.ConnectivityException, \
                    rostransform.ExtrapolationException):
                print("[INFO] Transformation not found")
                return
	    
            # TODO: use br.sendTransform (not necessary) 


    def get_object_depth(self, obj_struct, cv_image, depth_image, i):
  
            #
            X_SCALING = 1.333
            Y_SCALING = 0.75
            bbox = obj_struct.bbox
            top_left     = (int((bbox.center.x-(bbox.size_x/2))*X_SCALING)), \
                           (int((bbox.center.y-(bbox.size_y/2))*Y_SCALING))
            bottom_right = (int((bbox.center.x+(bbox.size_x/2))*X_SCALING)), \
                           (int((bbox.center.y+(bbox.size_y/2))*Y_SCALING))
	    center       = (int(bbox.center.x*X_SCALING)),(int(bbox.center.y*Y_SCALING)) 

	    # get average depth
	    RECT_ROI_DIST = 15
            # 
            if ( center[0]<RECT_ROI_DIST ):
                center[0] = RECT_ROI_DIST
            elif ( center[0]+RECT_ROI_DIST > IMAGE_WIDTH ):
                center[0] = IMAGE_WIDTH-RECT_ROI_DIST
            if ( center[1]<RECT_ROI_DIST ):
                center[1] = RECT_ROI_DIST
            elif ( center[1]+RECT_ROI_DIST > IMAGE_HEIGHT ):
                center[1] = IMAGE_HEIGHT-RECT_ROI_DIST
            #
            rect_roi_corners = [( center[0]-RECT_ROI_DIST, center[1]-RECT_ROI_DIST ), \
                                ( center[0]+RECT_ROI_DIST, center[1]+RECT_ROI_DIST)] 
	    rect_roi = depth_image[center[0]-RECT_ROI_DIST:center[0]+RECT_ROI_DIST, \
                                   center[1]-RECT_ROI_DIST:center[1]+RECT_ROI_DIST]
	    try:
                avg_depth = int(np.median(rect_roi))
            except:
                avg_depth = np.mean(rect_roi)
                pass
           
            avg_depth = depth_image[center[1],center[0]]

   
	    if DEBUG_MODE == 1:

                print('Object: '+str(i)+': '+str(center[0])+','+str(center[1])) 

                # write depth of object on images  
                cv2.putText(cv_image, str((avg_depth)), center, cv2.FONT_HERSHEY_SIMPLEX, \
                            1.0, (255,255,255), lineType=cv2.LINE_AA)
                cv2.putText(depth_image, str((avg_depth)), center, cv2.FONT_HERSHEY_SIMPLEX, \
                            1.0, (255,255,255), lineType=cv2.LINE_AA)

		# put coords on image 
                #cv2.putText(cv_image, str(top_left), top_left, cv2.FONT_HERSHEY_SIMPLEX, \ 
                #             1.0, (255,255,255), lineType=cv2.LINE_AA)
                #cv2.putText(cv_image, str(bottom_right), bottom_right, cv2.FONT_HERSHEY_SIMPLEX, \
                #             1.0, (255,255,255), lineType=cv2.LINE_AA)
                #cv2.putText(cv_image, str(bbox), (100,100), cv2.FONT_HERSHEY_SIMPLEX, \ 
                #             1.0, (255,255,255), lineType=cv2.LINE_AA)
	        #print(str(bbox))

                # draw bboxes on images
                cv2.rectangle(cv_image, top_left, bottom_right, (0,0,0), thickness=4)
                cv2.rectangle(cv_image, rect_roi_corners[0], rect_roi_corners[1], (0,0,0), thickness=4)
                cv2.circle(cv_image, center, 2, (255,255,255), -4)
                cv2.rectangle(depth_image, top_left, bottom_right, (0,0,0), thickness=4)
                cv2.rectangle(depth_image, rect_roi_corners[0], rect_roi_corners[1], (0,0,0), thickness=4)
                cv2.circle(depth_image, center, 2, (255,255,255), -4)

                # draw line from center of image to center of bbox
                image_center = (IMAGE_WIDTH/2, IMAGE_HEIGHT/2)
                cv2.circle(cv_image, image_center, 2, (255,255,255), -4)
                #
                cv2.line(cv_image, image_center, center, (255,0,0), 3)
                # center depth 
                center_depth = depth_image[image_center[1],image_center[0]]
                cv2.putText(cv_image, str((center_depth)), image_center, cv2.FONT_HERSHEY_SIMPLEX, \
                            1.0, (255,255,255), lineType=cv2.LINE_AA)
           
                signed_height = center[1] - IMAGE_HEIGHT/2 
                signed_width  = center[0] - IMAGE_WIDTH/2
                mid_height = (center[1]+(IMAGE_HEIGHT/2))/2
                mid_width  = (center[0]+(IMAGE_WIDTH/2))/2
                cv2.line(cv_image, (image_center[0],center[1]), image_center, (255,0,0), 3)
                cv2.line(cv_image, (image_center[0],center[1]), center, (255,0,0), 3)
         
                #cv2.putText(cv_image, 'u='+str((signed_height)), (mid_height,mid_width), \
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), lineType=cv2.LINE_AA) 
                #cv2.putText(cv_image, 'v='+str((signed_width)), (mid_height,mid_width), \
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), lineType=cv2.LINE_AA) 


            #if avg_depth == 0:
                #print('Object'+str(i)+'detected, but depth reads 0')
                 
 
	    return avg_depth, center, cv_image


    def object_predict(self, object_data, header, image_np,image):
        image_height,image_width,channels = image.shape
        obj=Detection2D()
        obj_hypothesis= ObjectHypothesisWithPose()

        object_id=object_data[0]
        object_score=object_data[1]
        dimensions=object_data[2]

        obj.header=header
        obj_hypothesis.id = object_id
        obj_hypothesis.score = object_score
        obj.results.append(obj_hypothesis)
        obj.bbox.size_y = int((dimensions[2]-dimensions[0])*image_height)
        obj.bbox.size_x = int((dimensions[3]-dimensions[1] )*image_width)
        obj.bbox.center.x = int((dimensions[1] + dimensions [3])*image_height/2)
        obj.bbox.center.y = int((dimensions[0] + dimensions[2])*image_width/2)

        #print(str(obj.bbox))
#        print('height:'+str(image_height))
#        print('width: '+str(image_width))

        return obj

def main(args):
    rospy.init_node('detector_node')
    obj=Detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("ShutDown")
    cv2.destroyAllWindows()

if __name__=='__main__':
    main(sys.argv)
