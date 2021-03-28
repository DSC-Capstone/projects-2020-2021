(function($){
  $(function(){

    $('.sidenav').sidenav();
    $('.parallax').parallax();

  }); // end of document ready
})(jQuery); // end of jQuery name space

var hidden_content_dict = {
  "nav_hidden_content1": "The data is collected using a ROS utility called <a href=http://wiki.ros.org/rosbag>rosbag</a>. Since the camera outputs images at 60FPS we throttle it down with the help of <a href=http://wiki.ros.org/topic_tools/throttle>topic_tools throttle</a> ROS package. This basically diverts the camera stream into a new topic that only outputs at a frequency of 1Hz. Once this is setup we run the throttle package to divert the input and use rosbag to record the diverted input giving us unique images across the track and reducing file size taken by a given lap. The images are then extracted from this bag file using the <a href=http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data>image_view</a> ROS package. Once converted into images we now have a part of our dataset ready",
  "nav_hidden_content2": "The dataset is labelled in the COCO JSON format with the help of <a href=https://www.makesense.ai/>MakeSenseAI</a>. We labelled around 2 laps of data. Since we ran a model that was pretrained on the COCO dataset we believed it would be sufficient. The images were cropped so that the top half is not visible during training and inference. This reduced a lot of the noise from the background",
  "nav_hidden_content3": "Training is done with the help of Facebook AI's <a href=https://github.com/facebookresearch/detectron2/>Detectron2</a> network. Specifically we use the MaskRCNN model in Detectron2 that has been pretrained on the COCO dataset. Using their libary allowed us to easily set up visualizations, inference and training. We noticed that a lower batch size of 8, initial learning rate of .01 and 200 epochs gave the best result. The scheduler reduced the learning rate soon after and the batch size of 8 led to faster convergence",
  "nav_hidden_content4": "The MaskRCNN model returns 4 objects: The boundary box which is the rectangular portion of the image where an object lies, the segmentation mask which is the exact pixels in which the object exists, the confidence level in the prediction and an integer ID representing what that object is. In our case the integer ID is mapped to our object strings: Lane and Cone",
  "nav_hidden_content5": "First we use the boundary boxes for the lanes. The great part about the boundary boxes is it perfectly contains the end points of the segmentation results. For the lanes we assume boundary boxes on the left are the left lane and boundary boxes on the right are the right lane (safe assumption if we are staying within the lanes at all times). The end points of those boundary boxes are connected to generate the base white polygon. If any cones appear in this white polygon we overlay the cone segmentation mask and blacken the area making it impermissible to move there. We then use the centroid of the image to move forward",
  "tuning_hidden_content1": "The collection of .bag files are then exported to .jpg files using the image_view ROS package. These exported files constitute our dataset. The main goal is to tune with different configuration settings to alleviate the amount of light sensitivity posed on the camera and make images as similar as possible to the regular, default settings. Configuration settings are saved as .yaml files within our workspace. We can modify our launch file to load the specific .yaml file that we want, whenever we run the car",
  "tuning_hidden_content2": "Different levels of parameters such as brightness, contrast, white balance, sharpness, hue etc. make qualitative changes to the image dataset collected during the daylight. For example, intensified and reduced light intensity is reflected on our images",
  "tuning_hidden_content3": "Lower levels of parameters generally showed a trend in tuning both metrics. In our experiment, lower white balance compensated for the “color cast” imposed by the light, and low sharpness level prevented the appearance of noise artifacts while emphasizing the content and detail of the image",
  "tuning_hidden_content4": "In contrast, the realtime performance evaluation between the default and non-tuned image set collected from daylight conditions gave us a similarity level of only 69%. Given these results, the best tuned configuration we found was proven to be valid for our mobile car driving on track across time",
  "mapping_hidden_content1": "The positional data consists of 8 columns, timestamp, x, y, z, pitch, roll, wheel angle, and yaw. We decided to discard timestamp, z, roll, and pitch columns. This is due to how our track has no elevation change and the car is only capable of movements on the yaw axis. Using the positional data, we are able to evaluate our map. Every time the car receives new image data, it creates key points on the image by placing yellow dots on them. These yellow dots are used to compares new images with older images. If there is a match, this forms a loop closure. A loop closure means that the car is able to localize itself at that current frame. In the image on the right, it shows that it was able to find multiple matching key points in two different images",
  "mapping_hidden_content2": "In order to get the true path, we used the middle yellow line of the track. The above image gave us an ATE of 0.46. However, after processing the map further in RTABMAP, we were able to find more loop closures and lower the ATE to 0.41",
  
}

function hidden_content(elem_id) {
  let button = document.getElementById(elem_id).children[0];
  let description = document.getElementById(elem_id+"_p");
  if (button.innerText.toLowerCase() == "arrow_downward") {
    button.innerText = "arrow_upward";
    description.innerHTML = hidden_content_dict[elem_id];
  }
  else {
    button.innerText = "arrow_downward";
    description.innerText = "";
  }
}


$(".dropdown-trigger").dropdown();