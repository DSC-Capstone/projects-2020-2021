#!/usr/bin/env python
import importlib
import math
import rospy
import genpy.message
from rospy import ROSException
import sensor_msgs.msg
from sensor_msgs.msg import Image, RegionOfInterest
from detectron2_ros.msg import Result
import actionlib
import rostopic
import rosservice
from threading import Thread
from rosservice import ROSServiceException
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np


class JoyTeleopException(Exception):
    pass

'''
Originally from https://github.com/ros-teleop/teleop_tools
Pulled on April 28, 2017.
Edited by Winter Guerra on April 28, 2017 to allow for default actions.
'''

class ResultsProcessor:
    def __init__(self):
        self.results_topic = "/masks"
        self.image_sub = rospy.Subscriber(self.results_topic, Result, \
            self.result_callback)
        self.latest_results = None


    def result_callback(self, data):
        """Transform ROS image to OpenCV image array, then save latest image"""
        self.latest_results = data

    def get_info(self):
        """Return latest image that has been converted to OpenCV image array"""
        return self.latest_results


class ObjectLaneAvoider:
    def __init__(self):
        self.y_center_offset = 0
        self.height = 360 - self.y_center_offset
        self.width = 1280
        self.QUEUE_SIZE = 5
        self.result_pub = rospy.Publisher('/obstacle_avoider/results',
                                      Image, queue_size = self.QUEUE_SIZE)
        self.process_pub = rospy.Publisher('/obstacle_avoider/process',
                                          Image, queue_size = self.QUEUE_SIZE)
        self.bridge_object = CvBridge()
        self.speed = 0.6
        self.steering_angle = 0  # straight
        self.WHEELBASE = 0.33

    def convert_angular_velocity_to_steering_angle(self, angular_velocity):
        if angular_velocity == 0:
            return 0
        return math.atan(angular_velocity * (self.WHEELBASE/self.speed))

    def draw_direction(self, final_mask, x, y):
        car_x, car_y = int(final_mask.shape[1]/2), final_mask.shape[0]
        cv2.line(final_mask, (car_x, car_y), (x, y), (255, 0, 0), 30)
        cv2.circle(final_mask, (x,y), 30, (0, 0, 255), -1)
        return final_mask

    def generate_lane_polygon(self, masks, bbox, lane_indices):
        best_left_box, best_right_box = self.find_best_box(masks, bbox, lane_indices)
        poly_coords = self.generating_box_coordinates( \
            best_left_box, best_right_box)
        poly_image = np.zeros((self.height, self.width))
        cv2.fillPoly(poly_image, pts = [np.array(poly_coords)], color=(255,255,255))
        return poly_image

    def find_best_box(self, masks, bbox, lane_indices):
        most_extreme_top_left = self.width
        most_extreme_top_right = 0
        best_left_box = None
        best_right_box = None
        for lane in lane_indices:
            # mask = self.bridge_object.imgmsg_to_cv2(masks[lane])[self.y_center_offset:]
            # coordinates = []
            # find_tops = np.diff(mask, axis = 0)
            # find_tops = np.where(find_tops == 0, -1, 255)
            # top_locations = np.argmax(find_tops, axis = 0)
            # for i, top in enumerate(top_locations):
            #     if find_tops[top][i] != -1:
            #         coordinates.append([top, i])
            # coordinates = np.array(coordinates)
            # slope = np.diff(coordinates, axis = 0)
            # num_rows = slope.shape[0]
            # running_slope = 0
            # for row in slope:
            #     if row[1] == 0:
            #         running_slope += row[0]
            #     else:
            #         running_slope += (row[0]/row[1])
            # slope = running_slope/num_rows
            current_bbox = bbox[lane]
            if (2 * current_bbox.x_offset + current_bbox.width)/2 < self.width/2:
                if current_bbox.x_offset < most_extreme_top_left:
                    best_left_box = current_bbox
                    most_extreme_top_left = current_bbox.x_offset
            else:
                if current_bbox.x_offset + current_bbox.width > most_extreme_top_right:
                    best_right_box = current_bbox
                    most_extreme_top_right = current_bbox.x_offset + current_bbox.width 
            # if slope > 0:
            #     if current_bbox.x_offset > most_extreme_top_right:
            #         most_extreme_top_right = current_bbox.x_offset
            #         best_right_box = current_bbox
            # else:
            #     if current_bbox.x_offset + current_bbox.width < most_extreme_top_left:
            #         most_extreme_top_left = current_bbox.x_offset + current_bbox.width
            #         best_left_box = current_bbox
        return best_left_box, best_right_box

    def generating_box_coordinates(self, best_left_box, best_right_box):
        if best_left_box is None and best_right_box is None:
            top_left = [0, 0]
            top_right = [self.width, 0]
            bottom_left = [0, self.height]
            bottom_right = [self.width, self.height]
        elif best_right_box is not None and best_left_box is not None:
            if best_right_box.y_offset > best_left_box.y_offset:
                highest_y_offset = best_right_box.y_offset
                box_height = best_right_box.height
            else:
                highest_y_offset = best_left_box.y_offset
                box_height = best_left_box.height
            highest_y_offset = max(0, highest_y_offset - self.y_center_offset)
            top_left = [best_left_box.x_offset + best_left_box.width, highest_y_offset]
            top_right = [best_right_box.x_offset, highest_y_offset]
            bottom_left = [best_left_box.x_offset, highest_y_offset + box_height]
            bottom_right = [best_right_box.x_offset + best_right_box.width, \
                highest_y_offset + box_height]
        elif best_left_box is not None:
            highest_y_offset = max(0, best_left_box.y_offset - self.y_center_offset)
            top_left = [best_left_box.x_offset + best_left_box.width, highest_y_offset]
            top_right = [self.width, highest_y_offset]
            bottom_left = [best_left_box.x_offset, highest_y_offset + best_left_box.height]
            bottom_right = [self.width, \
                highest_y_offset + best_left_box.height]
        else:
            highest_y_offset = max(0, best_right_box.y_offset - self.y_center_offset)
            top_left = [0, highest_y_offset]
            top_right = [best_right_box.x_offset, highest_y_offset]
            bottom_left = [0, highest_y_offset + best_right_box.height]
            bottom_right = [best_right_box.x_offset + best_right_box.width, \
                highest_y_offset + best_right_box.height]
        return [top_left, bottom_left, bottom_right, top_right]

    def generate_speed_steering(self, x, y):
        error_x = x - self.width/2
        angular_z = -error_x/100
        self.steering_angle = self.convert_angular_velocity_to_steering_angle(angular_z)
        self.steering_angle = np.clip(self.steering_angle, -0.4, 0.4)
        return self.speed, self.steering_angle

    def run(self, data):
        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(data.segmented_image)
            height, width, channels = cv_image.shape
        except CvBridgeError as e:
            print(e)
        cv_image = cv_image[self.y_center_offset:]
        class_names = np.array(data.class_names)
        lane_indices = np.where(class_names == 'lane')[0]
        lane_polygon = self.generate_lane_polygon(data.masks, \
            data.boxes, lane_indices)
        just_cone_data = np.zeros((self.height, self.width))
        for i, mask in enumerate(data.masks):
            if i not in lane_indices:
                temp = self.bridge_object.imgmsg_to_cv2(mask)[self.y_center_offset:]
                just_cone_data += temp
        just_cone_data = just_cone_data > 0
        joined_data = lane_polygon + just_cone_data
        joined_data[joined_data == 256] = 0
        joined_data[joined_data == 1] = 0
        joined_data[joined_data == 255] = 255
        m = cv2.moments(joined_data, False)
        try:
            x, y = m['m10']/m['m00'], m['m01']/m['m00']
        except ZeroDivisionError:
            x, y = self.width / 2, self.height / 2
        x = int(x)
        y = int(y)
        joined_data = np.repeat(joined_data.astype(np.uint8 \
            ).reshape(self.height, self.width, 1), 3, axis = 2)
        final_mask = self.draw_direction(joined_data, x, y)
        final_cv_image = self.draw_direction(self.bridge_object.imgmsg_to_cv2(data.segmented_image), x, y)
        self.process_pub.publish(self.bridge_object.cv2_to_imgmsg(final_mask))
        self.result_pub.publish(self.bridge_object.cv2_to_imgmsg(final_cv_image))
        return self.generate_speed_steering(x, y)
        

class JoyTeleop:
    """
    Generic joystick teleoperation node.
    Will not start without configuration, has to be stored in 'teleop' parameter.
    See config/joy_teleop.yaml for an example.
    """
    def __init__(self):
        if not rospy.has_param("teleop"):
            rospy.logfatal("no configuration was found, taking node down")
            raise JoyTeleopException("no config")

        self.publishers = {}
        self.al_clients = {}
        self.srv_clients = {}
        self.service_types = {}
        self.message_types = {}
        self.command_list = {}
        self.offline_actions = []
        self.offline_services = []

        self.old_buttons = []

        # custom line follower and image processing pipeline
        self.result_processor = ResultsProcessor()
        self.object_lane_avoider = ObjectLaneAvoider()

        teleop_cfg = rospy.get_param("teleop")

        for i in teleop_cfg:
            if i in self.command_list:
                rospy.logerr("command {} was duplicated".format(i))
                continue
            action_type = teleop_cfg[i]['type']
            self.add_command(i, teleop_cfg[i])
            if action_type == 'topic':
                self.register_topic(i, teleop_cfg[i])
            elif action_type == 'action':
                self.register_action(i, teleop_cfg[i])
            elif action_type == 'service':
                self.register_service(i, teleop_cfg[i])
            else:
                rospy.logerr("unknown type '%s' for command '%s'", action_type, i)

        # Don't subscribe until everything has been initialized.
        rospy.Subscriber('joy', sensor_msgs.msg.Joy, self.joy_callback)

        # Run a low-freq action updater
        rospy.Timer(rospy.Duration(2.0), self.update_actions)

    def joy_callback(self, data):
        try:
            for c in self.command_list:
                if self.match_command(c, data.buttons):
                    self.run_command(c, data)
                    # Only run 1 command at a time
                    break
        except JoyTeleopException as e:
            rospy.logerr("error while parsing joystick input: %s", str(e))
        self.old_buttons = data.buttons

    def register_topic(self, name, command):
        """Add a topic publisher for a joystick command"""
        topic_name = command['topic_name']
        try:
            topic_type = self.get_message_type(command['message_type'])
            self.publishers[topic_name] = rospy.Publisher(topic_name, topic_type, queue_size=1)
        except JoyTeleopException as e:
            rospy.logerr("could not register topic for command {}: {}".format(name, str(e)))

    def register_action(self, name, command):
        """Add an action client for a joystick command"""
        action_name = command['action_name']
        try:
            action_type = self.get_message_type(self.get_action_type(action_name))
            self.al_clients[action_name] = actionlib.SimpleActionClient(action_name, action_type)
            if action_name in self.offline_actions:
                self.offline_actions.remove(action_name)
        except JoyTeleopException:
            if action_name not in self.offline_actions:
                self.offline_actions.append(action_name)

    class AsyncServiceProxy(object):
        def __init__(self, name, service_class, persistent=True):
            try:
                rospy.wait_for_service(name, timeout=2.0)
            except ROSException:
                raise JoyTeleopException("Service {} is not available".format(name))
            self._service_proxy = rospy.ServiceProxy(name, service_class, persistent)
            self._thread = Thread(target=self._service_proxy, args=[])

        def __del__(self):
            # try to join our thread - no way I know of to interrupt a service
            # request
            if self._thread.is_alive():
                self._thread.join(1.0)

        def __call__(self, request):
            if self._thread.is_alive():
                self._thread.join(0.01)
                if self._thread.is_alive():
                    return False

            self._thread = Thread(target=self._service_proxy, args=[request])
            self._thread.start()
            return True

    def register_service(self, name, command):
        """ Add an AsyncServiceProxy for a joystick command """
        service_name = command['service_name']
        try:
            service_type = self.get_service_type(service_name)
            self.srv_clients[service_name] = self.AsyncServiceProxy(
                service_name,
                service_type)

            if service_name in self.offline_services:
                self.offline_services.remove(service_name)
        except JoyTeleopException:
            if service_name not in self.offline_services:
                self.offline_services.append(service_name)

    def match_command(self, c, buttons):
        """Find a command matching a joystick configuration"""
        # Buttons is a vector of the shape [0,1,0,1....
        # Turn it into a vector of form [1, 3...
        button_indexes = np.argwhere(buttons).flatten()

        # Check if the pressed buttons match the commands exactly.
        buttons_match = np.array_equal(self.command_list[c]['buttons'], button_indexes)

        #print button_indexes
        if buttons_match:
            return True

        # This might also be a default command.
        # We need to check if ANY commands match this set of pressed buttons.
        any_commands_matched = np.any([ np.array_equal(command['buttons'], button_indexes) for name, command in self.command_list.iteritems()])

        # Return the final result.
        return (buttons_match) or (not any_commands_matched and self.command_list[c]['is_default'])

    def add_command(self, name, command):
        """Add a command to the command list"""
        # Check if this is a default command
        if 'is_default' not in command:
            command['is_default'] = False

        if command['type'] == 'topic':
            if 'deadman_buttons' not in command:
                command['deadman_buttons'] = []
            command['buttons'] = command['deadman_buttons']
        elif command['type'] == 'action':
            if 'action_goal' not in command:
                command['action_goal'] = {}
        elif command['type'] == 'service':
            if 'service_request' not in command:
                command['service_request'] = {}
        self.command_list[name] = command

    def run_command(self, command, joy_state):
        """Run a joystick command"""
        cmd = self.command_list[command]

        if command == "autonomous_control":
            # new mode to detect autonomous control mode bound to RB
            self.run_auto_topic(command)
        elif cmd['type'] == 'topic':
            self.run_topic(command, joy_state)
        elif cmd['type'] == 'action':
            if cmd['action_name'] in self.offline_actions:
                rospy.logerr("command {} was not played because the action "
                             "server was unavailable. Trying to reconnect..."
                             .format(cmd['action_name']))
                self.register_action(command, self.command_list[command])
            else:
                if joy_state.buttons != self.old_buttons:
                    self.run_action(command, joy_state)
        elif cmd['type'] == 'service':
            if cmd['service_name'] in self.offline_services:
                rospy.logerr("command {} was not played because the service "
                             "server was unavailable. Trying to reconnect..."
                             .format(cmd['service_name']))
                self.register_service(command, self.command_list[command])
            else:
                if joy_state.buttons != self.old_buttons:
                    self.run_service(command, joy_state)
        else:
            raise JoyTeleopException('command {} is neither a topic publisher nor an action or service client'
                                     .format(command))

    def run_auto_topic(self, c):
        """Run command for autonomous mode."""
        cmd = self.command_list[c]
        msg = self.get_message_type(cmd["message_type"])()
        # rospy.logerr("Name: {}".format(c))

        # Do some result processing
        results = self.result_processor.get_info()
        throttle, steering = self.object_lane_avoider.run(results)

        # control car here
        self.set_member(msg, "drive.speed", throttle)
        self.set_member(msg, "drive.steering_angle", steering)

        self.publishers[cmd['topic_name']].publish(msg)

    def run_topic(self, c, joy_state):
        cmd = self.command_list[c]
        msg = self.get_message_type(cmd['message_type'])()
        if 'message_value' in cmd:
            for param in cmd['message_value']:
                self.set_member(msg, param['target'], param['value'])

        else:
            for mapping in cmd['axis_mappings']:
                if len(joy_state.axes)<=mapping['axis']:
                  rospy.logerr('Joystick has only {} axes (indexed from 0), but #{} was referenced in config.'.format(len(joy_state.axes), mapping['axis']))
                  val = 0.0
                else:
                  val = joy_state.axes[mapping['axis']] * mapping.get('scale', 1.0) + mapping.get('offset', 0.0)
                  # rospy.logerr("Testing: " + str(val))

                self.set_member(msg, mapping['target'], val)

        self.publishers[cmd['topic_name']].publish(msg)
        # rospy.logerr(msg)

    def run_action(self, c, joy_state):
        cmd = self.command_list[c]
        goal = self.get_message_type(self.get_action_type(cmd['action_name'])[:-6] + 'Goal')()
        genpy.message.fill_message_args(goal, [cmd['action_goal']])
        self.al_clients[cmd['action_name']].send_goal(goal)

    def run_service(self, c, joy_state):
        cmd = self.command_list[c]
        request = self.get_service_type(cmd['service_name'])._request_class()
        # should work for requests, too
        genpy.message.fill_message_args(request, [cmd['service_request']])
        if not self.srv_clients[cmd['service_name']](request):
            rospy.loginfo('Not sending new service request for command {} because previous request has not finished'
                          .format(c))

    def set_member(self, msg, member, value):
        ml = member.split('.')
        if len(ml) < 1:
            return
        target = msg
        for i in ml[:-1]:
            target = getattr(target, i)
        setattr(target, ml[-1], value)

    def get_message_type(self, type_name):
        if type_name not in self.message_types:
            try:
                package, message = type_name.split('/')
                mod = importlib.import_module(package + '.msg')
                self.message_types[type_name] = getattr(mod, message)
            except ValueError:
                raise JoyTeleopException("message type format error")
            except ImportError:
                raise JoyTeleopException("module {} could not be loaded".format(package))
            except AttributeError:
                raise JoyTeleopException("message {} could not be loaded from module {}".format(package, message))
        return self.message_types[type_name]

    def get_action_type(self, action_name):
        try:
            return rostopic._get_topic_type(rospy.resolve_name(action_name) + '/goal')[0][:-4]
        except TypeError:
            raise JoyTeleopException("could not find action {}".format(action_name))

    def get_service_type(self, service_name):
        if service_name not in self.service_types:
            try:
                self.service_types[service_name] = rosservice.get_service_class_by_name(service_name)
            except ROSServiceException, e:
                raise JoyTeleopException("service {} could not be loaded: {}".format(service_name, str(e)))
        return self.service_types[service_name]

    def update_actions(self, evt=None):
        for name, cmd in self.command_list.iteritems():
            if cmd['type'] != 'action':
                continue
            if cmd['action_name'] in self.offline_actions:
                self.register_action(name, cmd)


if __name__ == "__main__":
    try:
        rospy.init_node('dsc_180b_team_1_object_lane_avoider', anonymous = True)
        jt = JoyTeleop()
        rospy.spin()
    except JoyTeleopException:
        pass
    except rospy.ROSInterruptException:
        pass
