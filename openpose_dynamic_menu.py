"""
Wrap the OpenPose library with Python.
To install run `make install` and library will be stored in /usr/local/python
"""
import math
import subprocess

import speech_recognition as sr
from copy import deepcopy
import numpy as np
import ctypes as ct
import cv2
import os
from sys import platform
dir_path = os.path.dirname(os.path.realpath(__file__))

if platform == "win32":
    os.environ['PATH'] = dir_path + "/../../bin;" + os.environ['PATH']
    os.environ['PATH'] = dir_path + "/../../x64/Debug;" + os.environ['PATH']
    os.environ['PATH'] = dir_path + "/../../x64/Release;" + os.environ['PATH']

class OpenPose(object):
    """
    Ctypes linkage
    """
    if platform == "linux" or platform == "linux2":
        _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.so')
    elif platform == "darwin":
        _libop= np.ctypeslib.load_library('_openpose', dir_path+'/_openpose.dylib')
    elif platform == "win32":
        try:
            _libop= np.ctypeslib.load_library('_openpose', dir_path+'/Release/_openpose.dll')
        except OSError as e:
            _libop= np.ctypeslib.load_library('_openpose', dir_path+'/Debug/_openpose.dll')
    _libop.newOP.argtypes = [
        ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_float, ct.c_float, ct.c_int, ct.c_float, ct.c_int, ct.c_bool, ct.c_char_p]
    _libop.newOP.restype = ct.c_void_p
    _libop.delOP.argtypes = [ct.c_void_p]
    _libop.delOP.restype = None

    _libop.forward.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.uint8), ct.c_bool]
    _libop.forward.restype = None

    _libop.getOutputs.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.getOutputs.restype = None

    _libop.poseFromHeatmap.argtypes = [
        ct.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint8),
        ct.c_size_t, ct.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.uint8),
        np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.float32)]
    _libop.poseFromHeatmap.restype = None

    def encode(self, string):
        return ct.c_char_p(string.encode('utf-8'))

    def __init__(self, params):
        """
        OpenPose Constructor: Prepares OpenPose object

        Parameters
        ----------
        params : dict of required parameters. refer to openpose example for more details

        Returns
        -------
        outs: OpenPose object
        """
        self.op = self._libop.newOP(params["logging_level"],
		                            self.encode(params["output_resolution"]),
                                    self.encode(params["net_resolution"]),
                                    self.encode(params["model_pose"]),
                                    params["alpha_pose"],
                                    params["scale_gap"],
                                    params["scale_number"],
                                    params["render_threshold"],
                                    params["num_gpu_start"],
                                    params["disable_blending"],                    
                                    self.encode(params["default_model_folder"]))

    def __del__(self):
        """
        OpenPose Destructor: Destroys OpenPose object
        """
        self._libop.delOP(self.op)

    def forward(self, image, display = False):
        """
        Forward: Takes in an image and returns the human 2D poses, along with drawn image if required

        Parameters
        ----------
        image : color image of type ndarray
        display : If set to true, we return both the pose and an annotated image for visualization

        Returns
        -------
        array: ndarray of human 2D poses [People * BodyPart * XYConfidence]
        displayImage : image for visualization
        """
        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(3),dtype=np.int32)
        self._libop.forward(self.op, image, shape[0], shape[1], size, displayImage, display)
        array = np.zeros(shape=(size),dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        if display:
            return array, displayImage
        return array

    def poseFromHM(self, image, hm, ratios=[1]):
        """
        Pose From Heatmap: Takes in an image, computed heatmaps, and require scales and computes pose

        Parameters
        ----------
        image : color image of type ndarray
        hm : heatmap of type ndarray with heatmaps and part affinity fields
        ratios : scaling ration if needed to fuse multiple scales

        Returns
        -------
        array: ndarray of human 2D poses [People * BodyPart * XYConfidence]
        displayImage : image for visualization
        """
        if len(ratios) != len(hm):
            raise Exception("Ratio shape mismatch")

        # Find largest
        hm_combine = np.zeros(shape=(len(hm), hm[0].shape[1], hm[0].shape[2], hm[0].shape[3]),dtype=np.float32)
        i=0
        for h in hm:
           hm_combine[i,:,0:h.shape[2],0:h.shape[3]] = h
           i+=1
        hm = hm_combine

        ratios = np.array(ratios,dtype=np.float32)

        shape = image.shape
        displayImage = np.zeros(shape=(image.shape),dtype=np.uint8)
        size = np.zeros(shape=(4),dtype=np.int32)
        size[0] = hm.shape[0]
        size[1] = hm.shape[1]
        size[2] = hm.shape[2]
        size[3] = hm.shape[3]

        self._libop.poseFromHeatmap(self.op, image, shape[0], shape[1], displayImage, hm, size, ratios)
        array = np.zeros(shape=(size[0],size[1],size[2]),dtype=np.float32)
        self._libop.getOutputs(self.op, array)
        return array, displayImage

    @staticmethod
    def process_frames(frame, boxsize = 368, scales = [1]):
        base_net_res = None
        imagesForNet = []
        imagesOrig = []
        for idx, scale in enumerate(scales):
            # Calculate net resolution (width, height)
            if idx == 0:
                net_res = (16 * int((boxsize * frame.shape[1] / float(frame.shape[0]) / 16) + 0.5), boxsize)
                base_net_res = net_res
            else:
                net_res = ((min(base_net_res[0], max(1, int((base_net_res[0] * scale)+0.5)/16*16))),
                          (min(base_net_res[1], max(1, int((base_net_res[1] * scale)+0.5)/16*16))))
            input_res = [frame.shape[1], frame.shape[0]]
            scale_factor = min((net_res[0] - 1) / float(input_res[0] - 1), (net_res[1] - 1) / float(input_res[1] - 1))
            warp_matrix = np.array([[scale_factor,0,0],
                                    [0,scale_factor,0]])
            if scale_factor != 1:
                imageForNet = cv2.warpAffine(frame, warp_matrix, net_res, flags=(cv2.INTER_AREA if scale_factor < 1. else cv2.INTER_CUBIC), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            else:
                imageForNet = frame.copy()

            imageOrig = imageForNet.copy()
            imageForNet = imageForNet.astype(float)
            imageForNet = imageForNet/256. - 0.5
            imageForNet = np.transpose(imageForNet, (2,0,1))

            imagesForNet.append(imageForNet)
            imagesOrig.append(imageOrig)

        return imagesForNet, imagesOrig

    @staticmethod
    def draw_all(imageForNet, heatmaps, currIndex, div=4., norm=False):
        netDecreaseFactor = float(imageForNet.shape[0]) / float(heatmaps.shape[2]) # 8
        resized_heatmaps = np.zeros(shape=(heatmaps.shape[0], heatmaps.shape[1], imageForNet.shape[0], imageForNet.shape[1]))
        num_maps = heatmaps.shape[1]
        combined = None
        for i in range(0, num_maps):
            heatmap = heatmaps[0,i,:,:]
            resizedHeatmap = cv2.resize(heatmap, (0,0), fx=netDecreaseFactor, fy=netDecreaseFactor)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(resizedHeatmap)

            if i==currIndex and currIndex >=0:
                resizedHeatmap = np.abs(resizedHeatmap)
                resizedHeatmap = (resizedHeatmap*255.).astype(dtype='uint8')
                im_color = cv2.applyColorMap(resizedHeatmap, cv2.COLORMAP_JET)
                resizedHeatmap = cv2.addWeighted(imageForNet, 1, im_color, 0.3, 0)
                cv2.circle(resizedHeatmap, (int(maxLoc[0]),int(maxLoc[1])), 5, (255,0,0), -1)
                return resizedHeatmap
            else:
                resizedHeatmap = np.abs(resizedHeatmap)
                if combined is None:
                    combined = np.copy(resizedHeatmap);
                else:
                    if i <= num_maps-2:
                        combined += resizedHeatmap;
                        if norm:
                            combined = np.maximum(0, np.minimum(1, combined));

        if currIndex < 0:
            combined /= div
            combined = (combined*255.).astype(dtype='uint8')
            im_color = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
            combined = cv2.addWeighted(imageForNet, 0.5, im_color, 0.5, 0)
            cv2.circle(combined, (int(maxLoc[0]),int(maxLoc[1])), 5, (255,0,0), -1)
            return combined

# ----------------------------------------------------------------------------
#WARNING: Does not work with the latest openpose version which contains 25 joint positions, not 18
joint_names = ['nose', 'neck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'midhip', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear', 'lbigtoe', 'lsmalltoe', 'lheel', 'rbigtoe', 'rsmalltoe', 'rheel']

'''
Assign each of the tuples to a name that indicates which of the joints that tuple represents.

@param keypoint: the keypoint list from the .yml file
@return: a list of tuple for the joint name and the joint values
'''
def keypoints_to_joint_names(keypoints):
    global joint_names
	
    joints = []
    try:
        for i in range(0, len(keypoints)):
            joints.append([joint_names[i], keypoints[i]])
        return joints
    except:
        #If there are no joints in the frame, return an empty joint list for 1 person
        joints = []
        print("No Joints")
        for i in range(0, 25):
            joints.append([joint_names[i], [0,0,0]])
        return joints

'''
Get the joint values from the keypoint list based on which joint name was given as a parameter

@param keypoints: the keypoint data from the .yml file
@param joint_name: the name of the joint that the user wants to get the x,y, score valuess for
@return: a tuple of [x,y,score] for the given joint name
'''	
def get_keypoint_by_name(keypoints, joint_name):
	joints = keypoints_to_joint_names(keypoints)
	for name in joints:
		if name[0] == joint_name:
			return name[1]
	print("JOINT NOT FOUND")
	return False

'''
Method that draws the demo program given by EHU, which was originally in the __main__ code.

@param img: the image that the system must alter/work with.
@return: image that the system must print in the window of the program as output.
'''
def demo_program(keypoints):
    for i in range(len(keypoints)):
        person = keypoints[i]
        right_shoulder = get_keypoint_by_name(person, "rshoulder")
        #right_elbow = get_keypoint_by_name(person, "relbow")
        #right_wrist = get_keypoint_by_name(person, "rwrist")
        print(f"Person {i}: shoulder x: {right_shoulder[0]}, shoulder y: {right_shoulder[1]}, shoulder confidence: {right_shoulder[2]}")
       
# ===Start of code made by Yoran Kerbusch======================================    
global QUIT
global MAIN
global PROD
global SETT
global NEW
global LOAD
global current_mode

global menu_hold_min
        
QUIT = 0
MAIN = 1
PROD = 2
SETT = 3
NEW = 4
LOAD = 5
current_mode = MAIN

recogniser = sr.Recognizer()
background_voice_input = None
voice_input_text = None

#one time operation flags
newfilecopyflag = False
newfileflag = False
editorOpen = False

menus = [["varOne", "varTwo", "var", "if"], ["elif", "else", "while", "for"], ["continue", "break", "def", "and"], ["=", "==", "!=", ">"], ["<", "print", "new line", "tab"], ["delete line", "('", "')", "'"], ["1", "2", "3", "4"], ["5", "6", "7", "8"], ["9", "0", "space", "record input"], ["comma", "full stop", "run"]]
specific_menu = [0] * 5

ask_confirm = False

# Every inner array is [width, height]!
resolutions = [[1280, 1024, 0.85], [1366, 768, 0.95], [1440, 900, 1], [1600, 900, 1.1], [1920, 1080, 1.5]]
settings = {"mic_mute": False, "sound": True, "scale": 1.2, "distance": 2, "single": False, "resolution": 1}
changes_made = False
new_settings = deepcopy(settings)

work_file = None
files = []
files_pointer = 0
file_edited = False
FileEdit = os.path.join("User Scripts", "EDITING.py")

connections = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10], [10, 11], [11, 24], [12, 13], [13, 14], [14, 21], [15, 17], [16, 18], [19, 20], [21, 19], [21, 20], [22, 23], [24, 22], [24, 23]]
user_colors = [(225, 105, 65), (50, 205, 50), (0, 140, 255), (255, 0, 255), (42, 42, 165)]

menu_hold_min = 15
left_hand_count = [0] * 5
right_hand_count = [0] * 5

'''
Method that determines the amount of millimeters per pixel the camera is seeing, allowing for rudimentary distance approximation.
This is assuming the person is an adult and has an average distance between their eyes of around 63 millimeters.
@author Yoran Kerbusch (24143341)

@param person: array of np.ndarray is the person, represented by coordinates of joints of their body.
@param avg_eye_distance: int is the average distance adults eyes are apart, which is usually 63 millimeters.
@return distance: float is the amount of millimeters each pixel represents, assuming from an average distance between adult eyes.
'''
def eye_distance(person):
    distance = ""
    
    l_eye = get_keypoint_by_name(person, "leye")
    r_eye = get_keypoint_by_name(person, "reye")
    if ((l_eye[0] > 0.0) and (r_eye[0] > 0.0)):
        # Get the distance between the two eyes, if they are both visible.
        distance = math.sqrt((r_eye[0] - l_eye[0])**2 + (r_eye[1] - l_eye[1])**2)
        # Then calculate how much distance each millimeter will be, assuming on the average distance.
    else:
        distance = -1
    
    return distance

'''
Callback function that is used by the background microphones to let the menus know 
 when they picked up voice input, and what that voice input includes.
@author Yoran Kerbusch (24143341)

@param mic_recogniser: Recogniser is an instance of the SpeechRecogniser module, which can translate the raw data input that was collected.
@param raw_audio: AudioData the audio data the background microphone picked up.
@return translated: string is the translated/recognised version of the raw audio input of the user speaking.
'''
def universal_back_mic_callback(mic_recogniser, raw_audio):
    global background_voice_input
    
    translated = translate_raw_audio(mic_recogniser, raw_audio)
    
    print(translated)
    
    if (translated != "no_usable_input"):
        background_voice_input = translated

'''
Helper method that converts the raw audio given from the microphone to a space-separated string of words.
@author Yoran Kerbusch (24143341)

@param raw_audio: AudioData is the raw data picked up by the microphone, which has to be translated to be usable.
@return voice_result: string is the string picked up by the microphone from the raw input.
'''
def translate_raw_audio(mic_recogniser, raw_audio):
    try:
        voice_result = mic_recogniser.recognize_google(raw_audio)
        return voice_result
    except:
        voice_result = ("no_usable_input")
        return voice_result

'''
Helper method that can be used to calibrate the microhpone to the noise surrounding the user.
@author Yoran Kerbusch (24143341)

@param sample_seconds: int the amount of seconds the calibration should listen, to then use the background noise to establish a usable audio level to recogniser user input better.
'''
def calibrate_mic(mic, sample_seconds):
    global recogniser
    global background_voice_input
    
    # Use the standard input device of the computer at all times.
    with mic as source:
        # listen for 3 seconds and adjust to the ambient noise.
        recogniser.adjust_for_ambient_noise(source, sample_seconds)
        background_voice_input = None

'''
Method that starts a microphone that always listens on the background, listening for 
 phrases within a certain amount of seconds.
@author Yoran Kerbusch (24143341)

@param time_to_listen: int amount of seconds the microphone should listen for input.
@param calib_seconds: int amount of time before accepting input the system should take to calibrate for background noise.
@param stopper: func is a callback to stop a background listening microphone, which is given when a microphone is started.
'''
def start_background_mic(calib_seconds, callback):
    global recogniser
    global background_voice_input
    global voice_input_text
    
    # Listen to phrases of up to the given max amount of seconds, and do this forever (or until the menu is closed or stopped.)
    mic = sr.Microphone(device_index=1)
    
    #calibrate_mic(mic, calib_seconds)
    
    background_voice_input = None
    voice_input_text = None
    print("Microphone started!")
    return recogniser.listen_in_background(mic, callback)

'''
Method that stops the background microphone, if there is one currently being used.
It lets the calling system know if it succeeded by returning a boolean. True if the given stopper was used.
@author Yoran Kerbusch (24143341)

@param mic_stopper: func is a callback to stop a background listening microphone, which is given when a microphone is started.
@return succeeded: boolean is to let the calling system know if it was a valid stopper that was called.
'''
def stop_background_mic(mic_stopper):    
    global background_voice_input
    global voice_input_text
    
    if (mic_stopper != None):
        print("Microphone stopped!")
        print()
        mic_stopper(wait_for_stop=False)
        background_voice_input = None
        voice_input_text = None
        return True
    return False

'''
Method that can be used to get user input for a set amount of seconds, instead of listening on the background.
WARNING: If you have a microphone running on the background, use the method "timed_mic_input_stop_back" instead!
@author Yoran Kerbusch (24143341)

@param time_to_listen: int amount of seconds the microphone should listen for input.
@param calib_seconds: int amount of time before accepting input the system should take to calibrate for background noise.
@return voice_result: string containing the phrase the user has said, space separated for phrases.
'''
def timed_mic_input(time_to_listen, calib_seconds):
    global recogniser
    
    mic = sr.Microphone(device_index=1)
    
    #calibrate_mic(mic, calib_seconds)
    with mic as source:
        audio_input = recogniser.listen(source, time_to_listen, 2)
        
    return translate_raw_audio(recogniser, audio_input)

'''
Same method as "timed_mic_input", only this one requires a background microphone stopper as well.
This will stop the background microphone, listen for the timed input, and then restart the background microphone.
This is to prevent double input from the timed and background microhpone if they were both running.
@author Yoran Kerbusch (24143341)

@param time_to_listen: int amount of seconds the microphone should listen for input.
@param calib_seconds: int amount of time before accepting input the system should take to calibrate for background noise.
@return voice_result: string containing the phrase the user has said, space separated for phrases.
'''
def timed_mic_input_stop_back(time_to_listen, calib_seconds, back_mic_stopper, back_mic_callback):
    global recogniser
    global background_voice_input
    global voice_input_text
	
    stop_background_mic(back_mic_stopper)
    background_voice_input = None
    voice_input_text = None
    
    mic = sr.Microphone(device_index=1)
    
    #calibrate_mic(mic, calib_seconds)
    with mic as source:
        audio_input = recogniser.listen(source, time_to_listen, 2)
        
    voice_result = translate_raw_audio(recogniser, audio_input)
        
    stopper = start_background_mic(calib_seconds, back_mic_callback)
        
    return [voice_result, stopper]

'''
Method that displays the user's last voice input on the desired position, but only if the microphone is turned on.
@author Yoran Kerbusch (24143341)

@param img: The output image the text with the icon will be shown on to.
@param image_path: string is the path from the folder of this executable to the icon image for a microphone.
@param distance_x, distance_y: double is the distances from the top left corner of the screen, on which the scale will also be applied.
@param scale: double is the amount of units of the resolution should be applied upon the image to keep it away from borders, when 0 values are given from either distance.
'''
def mic_text(img, image_path, distance_x, distance_y, scale):
    global voice_input_text
    global resolutions
    global settings
    
    if (settings["mic_mute"] == False):
        scale_res = scale * resolutions[settings["resolution"]][2]
        distance_y = scale_res + (distance_y * resolutions[settings["resolution"]][2])
        distance_x = scale_res + (distance_x * resolutions[settings["resolution"]][2])
        
        s_img = cv2.imread(str(image_path))
        s_img = cv2.resize(s_img, (round((s_img.shape[0] / scale) * resolutions[settings["resolution"]][2]), round((s_img.shape[1] / scale) * resolutions[settings["resolution"]][2])))
        img[round(distance_y):round(distance_y + s_img.shape[1]), round(distance_x):round(distance_x + s_img.shape[0])] = s_img
        
        if (voice_input_text != None):
            cv2.putText(img, str(voice_input_text), (round(distance_x + s_img.shape[0]), round(distance_y + s_img.shape[1])), cv2.FONT_HERSHEY_TRIPLEX, (resolutions[settings["resolution"]][2]), (255, 255, 255), 1, cv2.LINE_AA, False)
        else:
            cv2.putText(img, "...", (round(distance_x + s_img.shape[0]), round(distance_y + s_img.shape[1])), cv2.FONT_HERSHEY_TRIPLEX, (resolutions[settings["resolution"]][2]), (255, 255, 255), 1, cv2.LINE_AA, False)

'''
Method that displays the user's last voice input on the desired position, but only if the microphone is turned on.
The difference is that this method allows for the usage of PNG images with transparent textures.
@author Yoran Kerbusch (24143341)

@param img: The output image the text with the icon will be shown on to.
@param image_path: string is the path from the folder of this executable to the icon image for a microphone.
@param distance_x, distance_y: double is the distances from the top left corner of the screen, on which the scale will also be applied.
@param scale: double is the amount of units of the resolution should be applied upon the image to keep it away from borders, when 0 values are given from either distance.
'''
def mic_text_alpha_icon(img, image_path, distance_x, distance_y, scale):
    global voice_input_text
    global resolutions
    global settings
    
    scale_res = scale * resolutions[settings["resolution"]][2]
    distance_y = scale_res + (distance_y * resolutions[settings["resolution"]][2])
    distance_x = scale_res + (distance_x * resolutions[settings["resolution"]][2])
    
    s_img = cv2.imread(str(image_path), -1)
    s_img = cv2.resize(s_img, (round((s_img.shape[0] / scale) * resolutions[settings["resolution"]][2]), round((s_img.shape[0] / scale) * resolutions[settings["resolution"]][2])))

    y1, y2 = round(scale_res + distance_y), round(scale_res + distance_y + s_img.shape[0])
    x1, x2 = round(scale_res + distance_x), round(scale_res + distance_x + s_img.shape[1])

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
        
    if (voice_input_text != None):
        cv2.putText(img, str(voice_input_text), (round(distance_x + s_img.shape[0]), round(distance_y + s_img.shape[1])), cv2.FONT_HERSHEY_TRIPLEX, (resolutions[settings["resolution"]][2]), (255, 255, 255), 1, cv2.LINE_AA, False)
    else:
        cv2.putText(img, "...", (round(distance_x + s_img.shape[0]), round(distance_y + s_img.shape[1])), cv2.FONT_HERSHEY_TRIPLEX, (resolutions[settings["resolution"]][2]), (255, 255, 255), 1, cv2.LINE_AA, False)


'''
Helper method to draw a line between two joints of a detected person, to draw the slightly
 transparent skeleton that gives them an idea of where they are in regards to the system.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the line on.
@param part1: nparray The first joint of the detected person, where we want to start the line.
@param part2: nparray The second joint of the detected person, where we want to end the line.
'''
def draw_connection(img, part1, part2):
    if isinstance(part1, np.ndarray) and isinstance(part2, np.ndarray):
        # Draw a transparent line between the (x,y) coordinates of the two given bodyparts.
        cv2.line(img, (part1[0], part1[1]), (part2[0], part2[1]), (96, 96, 96), 3)
        
'''
Helper method that draws a slightly transparent circle on the given joint on a person's joint.
If it's the nose, it'll draw it in the color of the person instead. If it's the wrists, it 
 draws the selection circle on there instead, again with a ring of the person's colour surrounding it.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the point on.
@param person: arraylist is a list of X & Y arrays that tell the system where each joint is located.
@param name: string is the name of the joint we are about to draw.
@param eye_dist: double is the distance of pixels between the user's eyes, which is used to determine how big the joint nodes should be drawn.
@param color: scalar of the numbers in the BGR configuration, dictating the colour of the joint node reflecting what user the person is.
@return part: nparray is the location of the joint on the screen OR False if the joint isn't valid to be drawn.
'''
def draw_joint(img, person, name, eye_dist, color):
    part = get_keypoint_by_name(person, name)
    if (part[0] > 0.0) and (part[1] > 0.0):
        if (name == "nose"):
            # Draw a transparent dot in the user's color on the given body part's x and y coordinates.
            cv2.circle(img, (part[0], part[1]), round(eye_dist / 6.5), color, -1)
        elif (name == "lwrist" or name == "rwrist"):
            cv2.circle(img, (part[0], part[1]), round(eye_dist / 4), color, -1)
            cv2.circle(img, (part[0], part[1]), round(eye_dist / 5.5), (201, 201, 201), -1)
        else:
            # Draw a transparent dot on the given body part's x and y coordinates.
            cv2.circle(img, (part[0], part[1]), round(eye_dist / 6.5), (95, 95, 95), -1)
            
        return part
    return False
        
'''
Method that draws a slightly transparent skeleton on top of the menus, to show the users where they are.
Some of the joints' nodes will be coloured to reflect each user, so they are recognisable.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the skeleton on.
@param person: arraylist is a list of X & Y arrays that tell the system where each joint is located.
@param eye_dist: double is the distance of pixels between the user's eyes, which is used to determine how big the joint nodes should be drawn.
@param color: scalar of the numbers in the BGR configuration, dictating the colour of the joint node reflecting what user the person is.
@return img: The output image with the user's slightly transparent skeleton now on it, to be outputted on the screen.
'''
def draw_skeleton(img, person, eye_dist, color):
    global joint_names
    global connections
    parts = []
    
    overlay = img.copy()
    
    # Draw transparent dots on each of the joints
    for part_name in joint_names:
        bodypart = draw_joint(overlay, person, part_name, eye_dist, color)
        parts.append(bodypart)
            
    # Then draw each of the connection lines between each joint that should be connected.
    for connection in connections:
        draw_connection(overlay, parts[connection[0]], parts[connection[1]])
    
    # Make the skeleton appear transparent, so it doesn't obstruct the buttons too much.
    img = cv2.addWeighted(overlay, 0.4, img, (1 - 0.4), 0, img)
    
    return img

'''
Helper method that draws a button with a text on a given position, in given colors and with a given text.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the button on.
@param x1, y1, x2, y2: int the positions of the top left point and the bottom right point of the rectangle that makes up the button, respectively.
@param block_color, text_color: scalar that describe in an BGR color of the button and the text within it, respectively.
@param text: string is the text that should be shown on the button.
@param originist: boolean tells the system wether the text should be drawn from the top-left or bottom-left. Default is False, drawing from the bottom-left.
@return button info: array is an array with the information of the button, including its positions and the text it displays.
'''
def draw_menu_block(img, x1, y1, x2, y2, block_color, text_color, text, originist=False):   
    global resolutions
    global settings
    
    cv2.rectangle(img, (x1, y1), (x2, y2), block_color, -1)
    
    if (originist):
        cv2.putText(img, text, ((x1 + 5), round(y1 + ((y2 - y1) / 1.5))), cv2.FONT_HERSHEY_TRIPLEX, round(resolutions[settings["resolution"]][2]), text_color, 1, cv2.LINE_AA, False)
    else:
        cv2.putText(img, text, (round(x1 + ((x2 - x1) / 8)), round(y1 + ((y2 - y1) / 1.5))), cv2.FONT_HERSHEY_TRIPLEX, round(resolutions[settings["resolution"]][2]), text_color, 1, cv2.LINE_AA, False)
    
    return [text, x1, y1, x2, y2]

'''
Method that checked if the person given is touching any of the buttons on screen. 
If so, it checks how much longer they have to hold it to select said button, filling the nodes on their wrists.
Upon selection, it returns the name of the selected button for this person.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown.
@param person: arraylist is a list of X & Y arrays that tell the system where each joint is located.
@param person_number: int is the number of the current person we're checking if they are selecting any buttons.
@param eye_dist: double is the distance of pixels between the user's eyes, which is used to determine how big the joint nodes should be drawn.
@param buttons: array a list of the coordinates of all the buttons selectable on the current shown menu.
@return buttons[i][0]: string identifier of the button the user has just selected, so the menu knows what to do.
'''
def menu_interaction(img, person, person_number, eye_dist, buttons): 
    global background_voice_input
    global voice_input_text
    global left_hand_count
    global right_hand_count
    
    # Get the wrists of the person to determine if they are pushing any buttons.
    l_wrist = get_keypoint_by_name(person, "lwrist")
    r_wrist = get_keypoint_by_name(person, "rwrist")
    
    # Keep track of if the person was touching a button this frame.
    l_was_touching = False
    r_was_touching = False
    
    for i in range(len(buttons)):
        if ((l_was_touching == True) and (r_was_touching == True)):
            # If we know both hands are touching but have not been held long enough yet, we don't have to check the other buttons.
            break
        
        if (((l_wrist[0] >= buttons[i][1]) and (l_wrist[0] <= buttons[i][3])) and ((l_wrist[1] >= buttons[i][2]) and (l_wrist[1] <= buttons[i][4]))):
            # If the left hand was pressing a button, then increase the timer and show it as an ellipse on the hand to show the progress.
            l_was_touching = True
            left_hand_count[person_number] = left_hand_count[person_number] + 1
            cv2.ellipse(img, (round(l_wrist[0]), round(l_wrist[1])), (round(eye_dist / 5), round(eye_dist / 5)), 0, 0, round(left_hand_count[person_number] * (360 / menu_hold_min)), (50, 205, 50), -1)
            
            if (left_hand_count[person_number] >= menu_hold_min):
                # If the left hand was touching for the minimum time, then tell the calling method that one of their menu buttons was selected.
                left_hand_count[person_number] = 0
                right_hand_count[person_number] = 0
                return buttons[i][0]
        
        if (((r_wrist[0] >= buttons[i][1]) and (r_wrist[0] <= buttons[i][3])) and ((r_wrist[1] >= buttons[i][2]) and (r_wrist[1] <= buttons[i][4]))):
            # If the left hand was pressing a button, then increase the timer and show it as an ellipse on the hand to show the progress.
            r_was_touching = True
            right_hand_count[person_number] = right_hand_count[person_number] + 1
            cv2.ellipse(img, (round(r_wrist[0]), round(r_wrist[1])), (round(eye_dist / 5), round(eye_dist / 5)), 0, 0, round(right_hand_count[person_number] * (360 / menu_hold_min)), (50, 205, 50), -1)
            
            if (right_hand_count[person_number] >= menu_hold_min):
                # If the right hand was touching for the minimum time, then tell the calling method that one of their menu buttons was selected.
                left_hand_count[person_number] = 0
                right_hand_count[person_number] = 0
                return buttons[i][0]
    
    if (l_was_touching == False):
        # If the left hand wasn't touching a button anymore, then reset the timer for it.
        left_hand_count[person_number] = 0
        
    if (r_was_touching == False):
        # If the right hand wasn't touching a button anymore, then reset the timer for it.
        right_hand_count[person_number] = 0
        
    if (background_voice_input != None):
        voice_input_text = background_voice_input
        background_voice_input = None
        return voice_input_text
        
'''
Method that draws all the contents of the confirmation pop-up of the system. This also controls what buttons should and 
 shouldn't be drawn, depending on settings and selected items on the menu, as instructed by the confirmation_menu_behaviour function.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the pop-up on.
@param w_scale, h_scale: double the amount of columns and rows respectively the screen should be cut up in. Used to change the size of the pop-up block.
@param question: string text that should be shown in the pop-up, to clarify to the user why the pop-up is shown.
@param positive, negative: string the options that should be shown to the user on the buttons.
@return confirmation_options: array is a list of the information of all the SELECTABLE buttons on the screen. This includes their text and positions.
'''
def draw_confirmation_menu(img, w_scale, h_scale, question, positive, negative):
    global resolutions
    global settings
    width = resolutions[settings["resolution"]][0]
    height = resolutions[settings["resolution"]][1]
    
    confirm_options = []
    
    overlay = img.copy()
    # Add a grey, slightly transparant overlay over the menu to make it clear to the user they can't select anything else at that moment.
    cv2.rectangle(overlay, (0, 0), (width, height), (63, 63, 63), -1)
    # Add that overlay on top of the existing menu.
    img = cv2.addWeighted(overlay, 0.85, img, (1 - 0.85), 0, img)
    
    box_start_x = round((width / w_scale))
    box_end_x = round((width / w_scale) * (w_scale - 1))
    
    box_start_y = round((height / h_scale) * 3)
    box_end_y = round((height / h_scale) * 6)
    
     # Calculate the size the text together with the buttons will take.
    res_scale = 6 * resolutions[settings["resolution"]][2]
    d_width = box_end_x - box_start_x
    button_width = round(d_width / 5)
    
    d_height = box_end_y - box_start_y
    button_height = d_height / 5
    
    # Add a solid box that will house the text and the two buttons.
    cv2.rectangle(img, (round(box_start_x - res_scale), round(box_start_y - res_scale)), (round(box_end_x + res_scale), round(box_end_y + res_scale)), (255, 255, 255), -1)
    cv2.rectangle(img, (box_start_x, box_start_y), (box_end_x, box_end_y), (0, 0, 0), -1)
    
    # Add the text with the query of the popup, originating from the top left of the box.
    cv2.putText(img, str(question), (round(box_start_x + res_scale), round(box_start_y + (res_scale * 5))), cv2.FONT_HERSHEY_TRIPLEX, resolutions[settings["resolution"]][2], (255, 255, 255), 1, cv2.LINE_AA, False)
    # Add the negative option button to the left bottom of the pop-up
    negative_button = draw_menu_block(img, round(box_start_x + res_scale), round(box_end_y - res_scale - button_height), (box_start_x + (button_width * 2)), round(box_end_y - res_scale), (127, 127, 127), (255, 255, 255), str(negative))
    negative_button[0] = "negative"
    confirm_options.append(negative_button)
    # Add the positive option button to the left bottom of the pop-up
    positive_button = draw_menu_block(img, (box_start_x + (button_width * 3)), round(box_end_y - res_scale - button_height), round(box_end_x - res_scale), round(box_end_y - res_scale), (127, 127, 127), (255, 255, 255), str(positive))
    positive_button[0] = "positive"
    confirm_options.append(positive_button)
    
    return confirm_options

'''
Method that controls what the system should do for the confirmation pop-up when one of the
 SELECTABLE buttons is selected by the user.
@author Yoran Kerbusch (24143341)

@param selected_menu: string is the identifier of the button the user just selected, so we know what we must do.
@return CHOICE: boolean lets the system that called the pop-up know if the user chose the positive or negative option in the pop-up.
'''
def confirmation_menu_behaviour(selected_menu, positive, negative):
    global ask_confirm
    
    selected = selected_menu.lower()
    
    # Set ask_confirm to False regardless, otherwise the pop-up would endlessly show.
    ask_confirm = False
    
    if (selected == "positive" or selected == positive.lower()):
        # If the user picks the positive option, they want to close the current mode and return to the previous mode
        return True
    elif (selected == "negative" or selected == negative.lower()):
        # If the user clicks the negative option, they want to return to the menu we are currently overlaying.
        return False
    return False
        
'''
Method that draws a pop-up over the current menu, which prompts the user(s) to undertake a certain action.
The pop-iup blocks them from using the menu until the users resolved it.
Chosing the positive option will return True, the negative option will return False. The calling menu needs to decide what to do with this.
THIS "MENU" MUST BE CALLED BY ANOTHER MENU, AS IT NEEDS A PLACE TO RETURN TO AFTER THE USER CHOSES EITHER OPTION!
@author Yoran Kerbusch (24143341)

@param openpose: The OpenPose object that we use to record and recognise the user's body and positions.
@param original_img: The image of the menu that was before this pop-up, so we can use that as a background.
@param w_scale, h_scale: double the amount of columns and rows respectively the screen should be cut up in. Used to change the size of the pop-up block.
@param question: string text that should be shown in the pop-up, to clarify to the user why the pop-up is shown.
@param positive, negative: string the options that should be shown to the user on the buttons.
@return CHOICE: boolean lets the system that called the pop-up know if the user chose the positive or negative option in the pop-up.
'''
def confirmation_menu(openpose, original_img, w_scale, h_scale, question, positive, negative):
    global ask_confirm
    global resolutions
    global settings
    
    cap = cv2.VideoCapture(0)
    
    mic_stopper = None
    if (settings["mic_mute"] == False):
        mic_stopper = start_background_mic(2, universal_back_mic_callback)
    
    while 1:
        img = original_img.copy()
        
        success, capture = cap.read()
        capture = cv2.resize(capture, (resolutions[settings["resolution"]][0], resolutions[settings["resolution"]][1]))
        
        keypoints = openpose.forward(capture, False)
        
        confirmation_menu = draw_confirmation_menu(img, w_scale, h_scale, question, positive, negative)
        
        person_limit = 5
        if (settings["single"] == True):
            person_limit = 1
    
        for i in range(min(len(keypoints), person_limit)):
            person = keypoints[i]
            eye_dist = eye_distance(person)
            
            # Use the color hex below with the actual colour of the person, reflecting which number user they are.
            img = draw_skeleton(img, person, eye_dist, user_colors[i])
            
            selected_menu = menu_interaction(img, person, i, eye_dist, confirmation_menu)
            if (selected_menu != None):
                stop_background_mic(mic_stopper)
                cap.release()
                return confirmation_menu_behaviour(selected_menu, positive, negative)
    
        cv2.imshow("output", img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            stop_background_mic(mic_stopper)
            cap.release()
            cv2.destroyAllWindows()
            ask_confirm = False
            return False

'''
Method that draws all the contents of the main menu of the system. This also controls what buttons should and 
 shouldn't be drawn, depending on settings and selected items on the menu, as instructed by the main_menu_behaviour function.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the menu on.
@param width, height: int values that control how big the input and output resolutions of the system are.
@param scale: double controls how much of the screen will be used width-wise.
@return main_options: array is a list of the information of all the SELECTABLE buttons on the screen. This includes their text and positions.
'''
def draw_main_menu(img, width, height, scale):
    global voice_input_text
    global resolutions
    global settings
    
    main_options = []
    
    # Draw the title of the current menu.
    cv2.putText(img, "Pynteractive Code Editor", (round(width / 5), 60), cv2.FONT_HERSHEY_TRIPLEX, round(2 * resolutions[settings["resolution"]][2]), (255, 255, 255), 2, cv2.LINE_AA, False)
    cv2.putText(img, "Adam Harvey & Yoran Kerbusch", (round((width / 6) * 2), 100), cv2.FONT_HERSHEY_TRIPLEX, round(resolutions[settings["resolution"]][2]), (255, 255, 255), 1, cv2.LINE_AA, False)
    
    # Draw the menu buttons fitting to the screen size available.
    start_x = round((width / scale) * 2)
    end_x = round((width / scale) * (scale - 2))
    
    button_space = (height - 100) / 9
    divider_size = button_space / 2
    button_size = button_space + divider_size
    
    next_button_origin = round(100 + divider_size)
    main_options.append(draw_menu_block(img, start_x, next_button_origin, end_x, round(next_button_origin + button_size), (127, 127, 127), (255, 255, 255), "New program"))
    
    next_button_origin = round(next_button_origin + button_size + divider_size)
    main_options.append(draw_menu_block(img, start_x, next_button_origin, end_x, round(next_button_origin + button_size), (127, 127, 127), (255, 255, 255), "Load program"))
    
    next_button_origin = round(next_button_origin + button_size + divider_size)
    main_options.append(draw_menu_block(img, start_x, next_button_origin, end_x, round(next_button_origin + button_size), (127, 127, 127), (255, 255, 255), "Settings"))
    
    next_button_origin = round(next_button_origin + button_size + divider_size)
    main_options.append(draw_menu_block(img, start_x, next_button_origin, end_x, round(next_button_origin + button_size), (127, 127, 127), (255, 255, 255), "Quit"))
    
    # Add text that shows the user what their last microphone input was.
    mic_text(img, "./Assets/white_mic.png", 0, 150, scale)
    
    return main_options

'''
Method that controls what the system should do for the main menu when one of the
 SELECTABLE buttons is selected by the user.
@author Yoran Kerbusch (24143341)

@param selected_menu: string is the identifier of the button the user just selected, so we know what we must do.
@return mode: int is the menu that the system should load next.
'''
def main_menu_behaviour(selected_menu):
    selected = selected_menu.lower()
    
    if ("new" in selected):
        return NEW
    elif ("load" in selected):
        return LOAD
    elif (selected == "settings"):
        return SETT 
    elif (selected == "quit"):
        return QUIT
    return -1

'''
Method that runs all the behaviour needed for the main menu, keeping the system in a
 loop here until the user selects an item to go to a different menu.
@author Yoran Kerbusch (24143341)

@param openpose: The OpenPose object that we use to record and recognise the user's body and positions.
@return MODE: int the mode to go to, but only when certain buttons are pressed, as dictated by the main_menu_behaviour() function.
'''
def main_menu(openpose):    
    global resolutions
    global settings
    
    cap = cv2.VideoCapture(0)
    
    mic_stopper = None
    if (settings["mic_mute"] == False):
        # If the microhpone is not muted, then start a background listener that listens at all times.
        mic_stopper = start_background_mic(2, universal_back_mic_callback)
    
    while 1:
        # Draw the menu items on a blank background with the buttons to go to the different modes.
        success, capture = cap.read()
        capture = cv2.resize(capture, (resolutions[settings["resolution"]][0], resolutions[settings["resolution"]][1]))
        
        # Code to have a blank background for the menu.
        img = np.zeros((resolutions[settings["resolution"]][1], resolutions[settings["resolution"]][0], 3), np.uint8)
        keypoints = openpose.forward(capture, False)
        
        # Code for using the live camera feed as the output.
        #keypoints, img = openpose.forward(capture, True)
        
        main_menu = draw_main_menu(img, resolutions[settings["resolution"]][0], resolutions[settings["resolution"]][1], 6);
        
        person_limit = 5
        if (settings["single"] == True):
            person_limit = 1
    
        for i in range(min(len(keypoints), person_limit)):
            person = keypoints[i]
            # Get the amount of pixels between the person's two eyes, if they are both visible.
            eye_dist = eye_distance(person)
            
            # Use the color hex below with the actual colour of the person, reflecting which number user they are.
            img = draw_skeleton(img, person, eye_dist, user_colors[i])
            
            selected_menu = menu_interaction(img, person, i, eye_dist, main_menu)
            if (selected_menu != None):
                # We need to stop the background microphone in case we go to a different menu.
                response = main_menu_behaviour(selected_menu)
                if (response >= 0):
                    # This means the user has held the button for long enough to select it.
                    cap.release()
                    # We need to stop the background microphone in case we go to a different menu.
                    stop_background_mic(mic_stopper)
                    return response
        
        cv2.imshow("output", img)
        
        # As a safe escape, if OpenPose stops working, the user can also return to the main menu by pressing the "esc" key on their keyboard.
        if cv2.waitKey(1) & 0xFF == 27:
            stop_background_mic(mic_stopper)
            cap.release()
            cv2.destroyAllWindows()
            return QUIT
   
'''
Helper method that generates a standard file name once the "new file" menu is selected.
File names will have a pattern of "py-in_script_X.py", where X is a number starting from 1.
The number will be changed to the next highest numbre if other files exist with the same name.
@author Yoran Kerbusch (24143341)

@return generated_name: string is the automatically generated name for the new file.
'''
def generate_file_name():
    files = [f for f in os.listdir('./User Scripts') if os.path.isfile(f)]
    generated_name = "py-in_script_"
    highest_value = 1
    
    for file in files:
        if (generated_name in file):
            # Remove the first part, then take all the digits that are now on the front.
            temp_string = file.replace(generated_name, "").replace(".py", "")
            # Then extract the numbers that should follow that standard string. If there are no numbers following it, then disregard this file.
            extracted_number = int(temp_string)
            if (extracted_number.isdigit()):
                # Check that we just got a single number, nothing more.
                if (extracted_number > highest_value):
                    # If the number of this previously system generated named file is higher than the current one we want to use, then use that older one, plus one. 
                    highest_value = (extracted_number + 1)
    
    return generated_name + str(highest_value) + ".py"
    
'''
Method that draws all the contents of the "new file" menu of the system. This also controls what buttons should and 
 shouldn't be drawn, depending on settings and selected items on the menu, as instructed by the new_menu_behaviour function.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the menu on.
@param width, height: int values that control how big the input and output resolutions of the system are.
@param scale: double controls how much of the screen will be used width-wise.
@param generated_name: string is the filename of the file that it would have if the user does not voice input a new one.
@return new_options: array is a list of the information of all the SELECTABLE buttons on the screen. This includes their text and positions.
'''
def draw_new_menu(img, width, height, scale, generated_name):
    global resolutions
    global settings
    global work_file
    
    new_options = []
    
    # Write the title for this menu down.
    cv2.putText(img, "New script", (round(width / 3), 60), cv2.FONT_HERSHEY_TRIPLEX, round(2 * resolutions[settings["resolution"]][2]), (255, 255, 255), 2, cv2.LINE_AA, False)
    
    # Get the sizes we have available to us for the current screen size.
    start_x = round((width / scale))
    end_x = round((width / scale) * (scale - 1))
    d_width = end_x - start_x
    
    space = (height - 75) / 8
    divider_size = space / 2
    button_size = space + divider_size
    
    # Draw the text box that shows the user the current name.
    next_button_origin = round(75 + (space * 1.5) + divider_size)
    draw_menu_block(img, start_x, next_button_origin, end_x, round(next_button_origin + button_size), (63, 63, 63), (127, 127, 127), "")
    draw_menu_block(img, (start_x + 6), (next_button_origin + 6), (end_x - 6), round((next_button_origin + button_size) - 6), (127, 127, 127), (255, 255, 255), work_file)

    # Draw the button that resets the name to the system generated name.
    next_button_origin = round(next_button_origin + button_size + divider_size)
    if (not generated_name == work_file):
        new_options.append(draw_menu_block(img, round(start_x + ((d_width / 6) * 2)), next_button_origin, round(end_x - ((d_width / 6) * 2)), round(next_button_origin + button_size), (127, 127, 127), (255, 255, 255), "Reset name"))
    else:
        draw_menu_block(img, round(start_x + ((d_width / 6) * 2)), next_button_origin, round(end_x - ((d_width / 6) * 2)), round(next_button_origin + button_size), (63, 63, 63), (127, 127, 127), "Reset name")
    
    bottom_space = width / 3
    third_bottom = bottom_space / 3
    bottom_button_start = round(height - ((height / 5.5) - divider_size))
    
    new_options.append(draw_menu_block(img, 0, bottom_button_start, round(third_bottom * 2), height, (127, 127, 127), (255, 255, 255), "Back"))
    if (not work_file == None):
        if (settings["mic_mute"] == False):
            # Only allow the user to rename files if their microphone is on. Otherwise, turn the button off.
            new_options.append(draw_menu_block(img, round(bottom_space + (third_bottom / 2)), bottom_button_start, round((bottom_space * 2) - (third_bottom / 2)), height, (127, 127, 127), (255, 255, 255), "Rename")) 
        else:
            draw_menu_block(img, round(bottom_space + (third_bottom / 2)), bottom_button_start, round((bottom_space * 2) - (third_bottom / 2)), height, (63, 63, 63), (127, 127, 127), "Rename") 
        
        new_options.append(draw_menu_block(img, round((bottom_space * 2) + third_bottom), bottom_button_start, width, height, (127, 127, 127), (255, 255, 255), "Create"))
    else:
        draw_menu_block(img, round(bottom_space + (third_bottom / 2)), bottom_button_start, round((bottom_space * 2) - (third_bottom / 2)), height, (63, 63, 63), (127, 127, 127), "Rename")
        draw_menu_block(img, round((bottom_space * 2) + third_bottom), bottom_button_start, width, height, (63, 63, 63), (127, 127, 127), "Create")
    
    mic_text(img, "./Assets/white_mic.png", 0, 150, scale)
    
    return new_options

'''
Method that controls what the system should do for the "new file" menu when one of the
 SELECTABLE buttons is selected by the user.
@author Yoran Kerbusch (24143341)

@param selected_menu: string is the identifier of the button the user just selected, so we know what we must do.
@param generated_name: string is the name of the file generated by the system upon loading the "new file" menu.
@return mode: int is the menu that the system should load next.
'''
def new_menu_behaviour(img, selected_menu, generated_name, mic_stopper=None):
    global work_file
    
    selected = selected_menu.lower()
    
    if (selected == "create"):
        # At this point, the name is either the auto-generated one, or one from voice input. All we have to do here is tell the production menu which file to open on the predefined path.
        return PROD
    elif (selected == "reset name"):
        # If the user presses this, they want to reset the voice input name they gave back to the system generated one.
        work_file = generated_name
    elif (selected == "rename"):
        s_img = cv2.imread("./Assets/white_mic.png")
        img_x = round((s_img.shape[0] / 2) * resolutions[settings["resolution"]][2])
        img_y = round((s_img.shape[1] / 2) * resolutions[settings["resolution"]][2])
        s_img = cv2.resize(s_img, (img_x, img_y))
        width = resolutions[settings["resolution"]][0]
        height = resolutions[settings["resolution"]][1]
        img[round(math.ceil((height / 2) - (img_y / 2))):round(math.ceil((height / 2) + (img_y / 2))), round(math.ceil((width / 2) - (img_x / 2))):round(math.ceil((width / 2) + (img_x / 2)))] = s_img
        cv2.imshow("output", img)
		
        # Listen for voice input, if the user has the mic on. If the mic is off, the button should not be available.
        if (mic_stopper != None):
            response = timed_mic_input_stop_back(10, 1, mic_stopper, universal_back_mic_callback)
            work_file = response[0].replace(" ", "_") + ".py"
            return response[1]
        else:
            work_file = timed_mic_input(10, 1).replace(" ", "_") + ".py"
    elif (selected == "back"):
        work_file == None
        return MAIN
    return -1
    
'''
Method that runs all the behaviour needed for the "new file" menu, keeping the system in a
 loop here until the user selects an item to go to a different menu.
@author Yoran Kerbusch (24143341)

@param openpose: The OpenPose object that we use to record and recognise the user's body and positions.
@return MODE: int the mode to go to, but only when certain buttons are pressed, as dictated by the new_menu_behaviour() function.
'''
def new_menu(openpose):    
    global resolutions
    global settings
    global work_file
    
    cap = cv2.VideoCapture(0)
    
    mic_stopper = None
    if (settings["mic_mute"] == False):
        mic_stopper = start_background_mic(2, universal_back_mic_callback)
    
    # Generate a name following standard naming conventions, since we need one if the user has the microphone off.
    generated_name = generate_file_name()
    work_file = generated_name
    
    while 1:
        success, capture = cap.read()
        capture = cv2.resize(capture, (resolutions[settings["resolution"]][0], resolutions[settings["resolution"]][1]))
        
        img = np.zeros((resolutions[settings["resolution"]][1], resolutions[settings["resolution"]][0], 3), np.uint8)
        keypoints = openpose.forward(capture, False)
        
        # Code for using the live camera feed as the output.
        #keypoints, img = openpose.forward(capture, True)
        
        new_menu = draw_new_menu(img, resolutions[settings["resolution"]][0], resolutions[settings["resolution"]][1], 6, generated_name);
        
        person_limit = 5
        if (settings["single"] == True):
            person_limit = 1
    
        for i in range(min(len(keypoints), person_limit)):
            person = keypoints[i]
            eye_dist = eye_distance(person)
         
            img = draw_skeleton(img, person, eye_dist, user_colors[i])
            
            selected_menu = menu_interaction(img, person, i, eye_dist, new_menu)
            if (selected_menu != None):
                response = new_menu_behaviour(img, selected_menu, generated_name, mic_stopper)
                if (isinstance(response, int) == False):
                    mic_stopper = response
                elif (response >= 0):
                    stop_background_mic(mic_stopper)
                    cap.release()
                    return response
        
        cv2.imshow("output", img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            stop_background_mic(mic_stopper)
            cap.release()
            cv2.destroyAllWindows()
            return MAIN
        
'''
Method that draws all the contents of the "load file" menu of the system. This also controls what buttons should and 
 shouldn't be drawn, depending on settings and selected items on the menu, as instructed by the load_menu_behaviour function.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the menu on.
@param width, height: int values that control how big the input and output resolutions of the system are.
@param scale: double controls how much of the screen will be used width-wise.
@param items: array is a list of the contents of the folder where all the files made with this system are put.
@return load_options: array is a list of the information of all the SELECTABLE buttons on the screen. This includes their text and positions.
'''
def draw_load_menu(img, width, height, scale, items):
    global resolutions
    global settings
    global work_file
    
    load_options = []
    
    cv2.putText(img, "Load script", (round(width / 3), 60), cv2.FONT_HERSHEY_TRIPLEX, round(2 * resolutions[settings["resolution"]][2]), (255, 255, 255), 2, cv2.LINE_AA, False)
    
    # Get the sizes available to us for this menu.
    start_x = round((width / scale))
    end_x = round((width / scale) * (scale - 1))
    d_width = end_x - start_x
    
    space = (height - 75) / 7
    divider_size = space / 2
    
    # Draw the buttons that allow the user to traverse the list, five items at a time.
    next_button_origin = round(75 + divider_size)
    load_options.append(draw_menu_block(img, start_x, (next_button_origin + 3), round(start_x + (d_width / scale)), ((next_button_origin * 5) - 3), (127, 127, 127), (255, 255, 255), "<"))
    load_options.append(draw_menu_block(img, round(end_x - (d_width / scale)), next_button_origin, end_x, ((next_button_origin * 5) - 3), (127, 127, 127), (255, 255, 255), ">"))
        
    # Draw the list of files found within the folder with all the python scripts made with this system.
    list_x_start = round(start_x + (d_width / scale))
    list_x_end = round(end_x - (d_width / scale))
    list_item_size = round((next_button_origin * 4) / 5)
    
    for i in (range(len(items))):
        if (not work_file == items[i]):
            load_options.append(draw_menu_block(img, (list_x_start + 6), (next_button_origin + 3), (list_x_end - 6), ((next_button_origin + list_item_size) - 3), (95, 95, 65), (223, 223, 223), items[i], True))
        else:
            # If the user has a file selected, highlight it in this list.
            draw_menu_block(img, (list_x_start + 6), (next_button_origin + 3), (list_x_end - 6), ((next_button_origin + list_item_size) - 3), (127, 127, 127), (255, 255, 255), items[i], True)
        next_button_origin = round(next_button_origin + list_item_size)
    
    bottom_space = width / 3
    third_bottom = bottom_space / 3
    bottom_button_start = round(height - ((height / 5.5) - divider_size))
    
    # Draw the bottom buttons to traverse between menus.
    load_options.append(draw_menu_block(img, 0, bottom_button_start, round(third_bottom * 2), height, (127, 127, 127), (255, 255, 255), "Back"))
    if (not work_file == None):
        if (settings["mic_mute"] == False):
            # Only allow the user to rename files if their microphone is on. Otherwise, turn the button off.
            load_options.append(draw_menu_block(img, round(bottom_space + (third_bottom / 2)), bottom_button_start, round((bottom_space * 2) - (third_bottom / 2)), height, (127, 127, 127), (255, 255, 255), "Rename")) 
        else:
            draw_menu_block(img, round(bottom_space + (third_bottom / 2)), bottom_button_start, round((bottom_space * 2) - (third_bottom / 2)), height, (63, 63, 63), (127, 127, 127), "Rename") 
        
        load_options.append(draw_menu_block(img, round((bottom_space * 2) + third_bottom), bottom_button_start, width, height, (127, 127, 127), (255, 255, 255), "Load"))
    else:
        draw_menu_block(img, round(bottom_space + (third_bottom / 2)), bottom_button_start, round((bottom_space * 2) - (third_bottom / 2)), height, (63, 63, 63), (127, 127, 127), "Rename")
        draw_menu_block(img, round((bottom_space * 2) + third_bottom), bottom_button_start, width, height, (63, 63, 63), (127, 127, 127), "Load")
    
    mic_text(img, "./Assets/white_mic.png", 0, 150, scale)
    
    return load_options

'''
Method that controls what the system should do for the "load file" menu when one of the
 SELECTABLE buttons is selected by the user.
@author Yoran Kerbusch (24143341)

@param selected_menu: string is the identifier of the button the user just selected, so we know what we must do.
@return mode: int is the menu that the system should load next.
'''
def load_menu_behaviour(img, selected_menu, mic_stopper):
    global files_pointer
    global work_file
    
    selected = selected_menu.lower()
    
    if (selected == ">" or selected == "next"):
        # Go to the next page of the list of files. If there are no more, just go to the first page.
        files_pointer = files_pointer + 5
        if (files_pointer > (len(files) - 1)):
            files_pointer = 0
    elif(selected == "<" or selected == "previous"):
        # Go to the previous page of the list of files. If there are no more, just go to the last page.
        files_pointer = files_pointer - 5
        if (files_pointer < 0):
            files_pointer = len(files) - (len(files) % 5)
    elif (selected == "load"):
        # Load the currently selected file.
        files_pointer = 0
        return PROD
    elif (selected == "rename"):
        s_img = cv2.imread("./Assets/white_mic.png")
        img_x = round((s_img.shape[0] / 2) * resolutions[settings["resolution"]][2])
        img_y = round((s_img.shape[1] / 2) * resolutions[settings["resolution"]][2])
        s_img = cv2.resize(s_img, (img_x, img_y))
        width = resolutions[settings["resolution"]][0]
        height = resolutions[settings["resolution"]][1]
        img[round(math.ceil((height / 2) - (img_y / 2))):round(math.ceil((height / 2) + (img_y / 2))), round(math.ceil((width / 2) - (img_x / 2))):round(math.ceil((width / 2) + (img_x / 2)))] = s_img
        cv2.imshow("output", img)	
	
        # Allow the user to rename the selected file, if their microphone is on.										
        new_name = work_file
        new_stopper = None
        if (mic_stopper != None):
            response = timed_mic_input_stop_back(10, 1, mic_stopper, universal_back_mic_callback)
            new_name = response[0].replace(" ", "_") + ".py"
            new_stopper = response[1]
        else:
            new_name = timed_mic_input(10, 1).replace(" ", "_") + ".py"
        
		# Actually replace the name of the file in the directory of the computer...
        file_index = files.index(work_file)
        os.rename(os.getcwd() + "/User Scripts/" + files[file_index], os.getcwd() + "/User Scripts/" + new_name)
        files[file_index] = new_name
        work_file = new_name
		
        if (new_stopper != None):
            return new_stopper
    elif(".py" in selected):
        if (work_file == selected_menu):
            # Deselect the currently selected file if it is selected again.
            work_file = None
        else:
            # If a file is selected, then set the file to load to that one.
            work_file = selected_menu
    elif (selected == "back"):
        work_file = None
        files_pointer = 0
        return MAIN
    return -1
        
'''
Method that runs all the behaviour needed for the "load file" menu, keeping the system in a
 loop here until the user selects an item to go to a different menu.
@author Yoran Kerbusch (24143341)

@param openpose: The OpenPose object that we use to record and recognise the user's body and positions.
@return MODE: int the mode to go to, but only when certain buttons are pressed, as dictated by the load_menu_behaviour() function.
'''
def load_menu(openpose):
    global resolutions
    global settings
    global files
    global files_pointer
    
    cap = cv2.VideoCapture(0)
    
    mic_stopper = None
    if (settings["mic_mute"] == False):
        mic_stopper = start_background_mic(2, universal_back_mic_callback)
    
    files = [f for f in os.listdir('./User Scripts') if ".py" in f]
    
    while 1:
        success, capture = cap.read()
        capture = cv2.resize(capture, (resolutions[settings["resolution"]][0], resolutions[settings["resolution"]][1]))
    
        img = np.zeros((resolutions[settings["resolution"]][1], resolutions[settings["resolution"]][0], 3), np.uint8)
        keypoints = openpose.forward(capture, False)
        
        # Code for using the live camera feed as the output.
        #keypoints, img = openpose.forward(capture, True)
        
        end_slice = min((len(files)), (files_pointer + 5))
        load_menu = draw_load_menu(img, resolutions[settings["resolution"]][0], resolutions[settings["resolution"]][1], 8, files[files_pointer:end_slice]);
        
        person_limit = 5
        if (settings["single"] == True):
            person_limit = 1
    
        for i in range(min(len(keypoints), person_limit)):
            person = keypoints[i]
            eye_dist = eye_distance(person)
      
            img = draw_skeleton(img, person, eye_dist, user_colors[i])
            
            selected_menu = menu_interaction(img, person, i, eye_dist, load_menu)
            if (selected_menu != None):
                response = load_menu_behaviour(img, selected_menu, mic_stopper)
                if (isinstance(response, int) == False):
                    mic_stopper = response
                elif (response >= 0):
                    stop_background_mic(mic_stopper)
                    cap.release()
                    return response
        
        cv2.imshow("output", img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            stop_background_mic(mic_stopper)
            cap.release()
            cv2.destroyAllWindows()
            return MAIN
        
'''
Helper method that draws a horizontal option block. This has an off & on button, which can be toggled between. 
The method will control which of the two buttons is highlighted, the other then being selectable to switch.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the toggle buttons on.
@param option_text: string is the text shown to the user next to the toggle, so they know what they are selecting.
@param setting_name: string is the identifier of the setting this controls. This is so the system can easily know what toggle has been pressed.
@param origin_x, origin_y, end_x, end_y: int are the coordinates of the complete space this toggle section can occupy on screen.
@return button: array is the name and coordinates of the button on the toggle that should be selectable.
'''
def toggle_option(img, option_text, setting_name, origin_x, origin_y, end_x, end_y):
    d_x = end_x - origin_x
    text_box_end = round(origin_x + (d_x / 2.5))
    first_button_end = round((d_x / 2.85) + text_box_end)
    
    # Draw the box with the text of the setting.
    draw_menu_block(img, origin_x, origin_y, text_box_end, end_y, (159, 159, 159), (255, 255, 255), option_text)
    if (new_settings[setting_name] == True):
        # Make the "On" button not selectable, as that's what the setting is currently set to.
        draw_menu_block(img, round((d_x / 10) + text_box_end), origin_y, first_button_end, end_y, (127, 127, 127), (255, 255, 255), "On")
        
        # So make the "Off" button grey and selectable, so the user knows this is not turned on.
        button = draw_menu_block(img, first_button_end, origin_y, end_x, end_y, (63, 63, 63), (127, 127, 127), "Off")
        button[0] = setting_name
        
        return button
    else:
        # The setting is off, so make the "On" button grey and selectable, so that the user can turn this on.
        button = draw_menu_block(img, round((d_x / 10) + text_box_end), origin_y, first_button_end, end_y, (63, 63, 63), (127, 127, 127), "On")
        button[0] = setting_name
        
        # And make the "Off" button bright and not selectable, since this setting is already off.
        draw_menu_block(img, first_button_end, origin_y, end_x, end_y, (127, 127, 127), (255, 255, 255), "Off")
        
        return button
        
'''
Helper method that adds a settings selection to increase/decrease a numeral value. This has an lower and upper limit.
If one of these limits is reached, this method will turn the button to further lower or decrease it off, so the user can't go over the respective limit.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the toggle buttons on.
@param option_text: string is the text shown to the user next to the toggle, so they know what they are selecting.
@param setting_name: string is the identifier of the setting this controls. This is so the system can easily know what toggle has been pressed.
@param lower_limit, upper_limit: double are the lower and upper bounds that this integer increase/decrease button should allow the user to go up to.
@param origin_x, origin_y, end_x, end_y: int are the coordinates of the complete space this toggle section can occupy on screen.
@return selectable_buttons: array of arrays with the text and coordinates of each of the buttons currently selectable on this increase/decrease selector.
'''
def alter_option(img, option_text, setting_name, lower_limit, upper_limit, origin_x, origin_y, end_x, end_y):
    selectable_buttons = []
    
    d_x = end_x - origin_x
    text_box_end = round(origin_x + (d_x / 2.5))
    selector_unit = (d_x / 2.5) / 5
    
    # Draw the box with the text of the setting.
    draw_menu_block(img, origin_x, origin_y, text_box_end, end_y, (159, 159, 159), (255, 255, 255), option_text)
    
    text_box_end = round((d_x / 10) + text_box_end)
    
    if (new_settings[setting_name] > lower_limit):
        # Draw the button as a selectable one if the upper limit has not been reached yet.
        button = draw_menu_block(img, text_box_end, origin_y, round(text_box_end + selector_unit), end_y, (127, 127, 127), (255, 255, 255), "-")
        button[0] = setting_name + " down"
        selectable_buttons.append(button)
    else:
        # Otherwise, don't append it and make it darker, making it clear it isn't selectable.
        draw_menu_block(img, text_box_end, origin_y, round(text_box_end + selector_unit), end_y, (63, 63, 63), (127, 127, 127), "-")
    
    if (setting_name == "resolution"):
        # If the setting is specifically for the resolution, we want to show the resolution values, instead of the pointer that is actually stored in the settings array.
        draw_menu_block(img, round(text_box_end + selector_unit), origin_y, round(text_box_end + (selector_unit * 5.25)), end_y, (255, 255, 255), (0, 0, 0), str(resolutions[new_settings[setting_name]][0]) + "x" + str(resolutions[new_settings[setting_name]][1]))
    else:
        # For the rest of the settings, show their direct values as stored in the settings array.
        draw_menu_block(img, round(text_box_end + selector_unit), origin_y, round(text_box_end + (selector_unit * 5.25)), end_y, (255, 255, 255), (0, 0, 0), str(new_settings[setting_name]))
    
    if (new_settings[setting_name] < upper_limit):
        # Draw the button as a selectable one if the upper limit has not been reached yet.
        button = draw_menu_block(img, round(text_box_end + (selector_unit * 5.25)), origin_y, end_x, end_y, (127, 127, 127), (255, 255, 255), "+")
        button[0] = setting_name + " up"
        selectable_buttons.append(button)
    else:
        # Otherwise, don't append it and make it darker, making it clear it isn't selectable.
        draw_menu_block(img, round(text_box_end + (selector_unit * 5.25)), origin_y, end_x, end_y, (63, 63, 63), (127, 127, 127), "+")
    
    return selectable_buttons
    
'''
Method that draws all the contents of the settings menu of the system. This also controls what buttons should and 
 shouldn't be drawn, depending on settings and selected items on the menu, as instructed by the settings_menu_behaviour function.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the menu on.
@param width, height: int values that control how big the input and output resolutions of the system are.
@param scale: double controls how much of the screen will be used width-wise.
@return settings_options: array is a list of the information of all the SELECTABLE buttons on the screen. This includes their text and positions.
'''
def draw_settings_menu(img, width, height, scale):
    global resolutions
    global settings
    global changes_made
    
    settings_buttons = []
    
    cv2.putText(img, "Settings", (round(width / 2.5), 60), cv2.FONT_HERSHEY_TRIPLEX, round(2 * resolutions[settings["resolution"]][2]), (255, 255, 255), 2, cv2.LINE_AA, False)
    
    # Draw all the different toggles and alters to change the settings.
    start_x = round((width / scale))
    end_x = round((width / scale) * (scale - 1))
    
    space = (height - 75) / 12
    divider_size = space / 2
    button_size = space + divider_size
    
    next_button_origin = round(75 + divider_size)
    settings_buttons.append(toggle_option(img, "Mute microphone", "mic_mute", start_x, next_button_origin, end_x, round(next_button_origin + button_size)))
        
    next_button_origin = round(next_button_origin + button_size + divider_size)
    settings_buttons = settings_buttons + alter_option(img, "Button size scale", "scale", 1, 15, start_x, next_button_origin, end_x, round(next_button_origin + button_size))
    
    next_button_origin = round(next_button_origin + button_size + divider_size)
    settings_buttons = settings_buttons + alter_option(img, "Distance scale", "distance", 1, 15, start_x, next_button_origin, end_x, round(next_button_origin + button_size))
    
    next_button_origin = round(next_button_origin + button_size + divider_size)
    settings_buttons.append(toggle_option(img, "Single-user mode", "single", start_x, next_button_origin, end_x, round(next_button_origin + button_size)))
    
    next_button_origin = round(next_button_origin + button_size + divider_size)
    settings_buttons = settings_buttons + alter_option(img, "Resolution", "resolution", 0, (len(resolutions) - 1), start_x, next_button_origin, end_x, round(next_button_origin + button_size))

    # Draw the bottom three buttons that navigate between the menus.
    bottom_space = width / 3
    third_bottom = bottom_space / 3
    
    settings_buttons.append(draw_menu_block(img, 0, round(height - button_size), round(third_bottom * 2), height, (127, 127, 127), (255, 255, 255), "Back"))
    if (changes_made):
        # Only if changes were made should the save button be interactable & brighter.
        settings_buttons.append(draw_menu_block(img, round((bottom_space * 2) + third_bottom), round(height - button_size), width, height, (127, 127, 127), (255, 255, 255), "Save changes"))
    else:
        # If no settings were changed, make the save button darker and don't return it, so it isn't interactable.
        draw_menu_block(img, round((bottom_space * 2) + third_bottom), round(height - button_size), width, height, (63, 63, 63), (127, 127, 127), "Save changes")
    
    mic_text(img, "./Assets/white_mic.png", 0, 75, scale)
    
    return settings_buttons

'''
Method that controls what the system should do for the settings menu when one of the
 SELECTABLE buttons is selected by the user.
@author Yoran Kerbusch (24143341)

@param selected_menu: string is the identifier of the button the user just selected, so we know what we must do.
@param previous_mode: int is the integer identifier of the mode the system was in before going to the settings menu.
@return mode: int is the menu that the system should load next.
'''
def settings_menu_behaviour(selected_menu, previous_mode):
    global settings
    global changes_made
    global new_settings
    global ask_confirm
    
    selected = selected_menu.lower()
    
    if ("mute" in selected):
        # Change the user's setting for this option
        new_settings["mic_mute"] = not new_settings["mic_mute"]
        changes_made = True
    elif (selected == "scale up"  or selected == "scale app"):
        # Increase the scaling for button size as the user pressed that button.
        new_settings["scale"] = round((new_settings["scale"] + 0.1), 1)
        changes_made = True
    elif (selected == "scale down"):
        # Decrease the scaling for button size as the user pressed that button.
        new_settings["scale"] = round((new_settings["scale"] - 0.1), 1)
        changes_made = True
    elif (selected == "distance up" or selected == "distance app"):
        # Increase the distance the buttons will be from the user in production mode.
        new_settings["distance"] = round((new_settings["distance"] + 0.1), 1)
        changes_made = True
    elif (selected == "distance down"):
        # Decrease the distance the buttons will be from the user in production mode.
        new_settings["distance"] = round((new_settings["distance"] - 0.1), 1)
        changes_made = True
    elif ("single" in selected):
        # Change the user's setting for this option
        new_settings["single"] = not new_settings["single"]
        changes_made = True
    elif (selected == "resolution up" or selected == "resolution up"):
        # Change the resolution of the window to the next higher resolution.
        new_settings["resolution"] = new_settings["resolution"] + 1
        changes_made = True
    elif (selected == "resolution down"):
        # Change the resolution of the window to the next lower resolution.
        new_settings["resolution"] = new_settings["resolution"] - 1
        changes_made = True
    elif (selected == "save changes"):
        # Save & apply the changes the user has made to the settings.
        settings = deepcopy(new_settings)
        changes_made = False
    elif (selected == "back"):
        # Return to the previous menu the user was at.
        if (changes_made == True):
            # If the user made changes to the settings without saving them, ask them if they are sure they want to go back.
            ask_confirm = True
            return -1
        return previous_mode
    return -1
    
'''
Method that runs all the behaviour needed for the settings menu, keeping the system in a
 loop here until the user selects an item to go to a different menu.
@author Yoran Kerbusch (24143341)

@param openpose: The OpenPose object that we use to record and recognise the user's body and positions.
@param previous_mode: int the previous mode the user was in, before going to the settings menu. This is so we can return back to that afterwards.
@return MODE: int the mode to go to, but only when certain buttons are pressed, as dictated by the settings_menu_behaviour() function.
'''
def settings_menu(openpose, previous_mode):    
    global resolutions
    global settings
    global new_settings
    global changes_made
    global ask_confirm
    
    cap = cv2.VideoCapture(0)
    
    mic_stopper = None
    if (settings["mic_mute"] == False):
        mic_stopper = start_background_mic(2, universal_back_mic_callback)
    
    while 1:
        success, capture = cap.read()
        capture = cv2.resize(capture, (resolutions[settings["resolution"]][0], resolutions[settings["resolution"]][1]))
        
        img = np.zeros((resolutions[settings["resolution"]][1], resolutions[settings["resolution"]][0], 3), np.uint8)
        keypoints = openpose.forward(capture, False)
        
        # Code for using the live camera feed as the output.
        #keypoints, img = openpose.forward(capture, True)
        
        settings_menu = draw_settings_menu(img, resolutions[settings["resolution"]][0], resolutions[settings["resolution"]][1], 6);
        
        if (ask_confirm == True):
            # Do the confirmation pop-up before checks are done on drawing people, so the background isn't cluttered.
            stop_background_mic(mic_stopper)
            cap.release()
            response = confirmation_menu(openpose, img, 6, 9, "Return without saving changes to settings?", "Yes", "Back to settings")
            if (response == True):
                # The user does not want to save their changes made, thus return to the previous menu.
                new_settings = deepcopy(settings)
                ask_confirm = False
                changes_made = False
                return previous_mode
            # Otherwise, do nothing, as the user wants to continue in the settings menu.
            cap = cv2.VideoCapture(0)
            mic_stopper = start_background_mic(2, universal_back_mic_callback)
            continue
        
        person_limit = 5
        if (settings["single"] == True):
            person_limit = 1
    
        for i in range(min(len(keypoints), person_limit)):
            person = keypoints[i]
            eye_dist = eye_distance(person)
            
            img = draw_skeleton(img, person, eye_dist, user_colors[i])
            
            selected_menu = menu_interaction(img, person, i, eye_dist, settings_menu)
            if (selected_menu != None):
                response = settings_menu_behaviour(selected_menu, previous_mode)
                if (response >= 0):
                    stop_background_mic(mic_stopper)
                    cap.release()
                    return response
        
        cv2.imshow("output", img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            stop_background_mic(mic_stopper)
            cap.release()
            cv2.destroyAllWindows()
            return previous_mode
        
'''
Method that draws the top bar of buttons to the production menu, allowing the user
 to go to some of the different big menus or save their work.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the menu on.
@param width, height: int the dimensions of the full production view the users see.
@param resolution_scale: double the value to scale buttons and texts by for the user's current used screen resolution.
@param file_edited: boolean tells the method if the file has changed. If so, it will enable the save button.
@return static_options: array of arrays including the coordinates of all the buttons.
'''
def draw_static_prod_menu(img, width, height, resolution_scale, file_edited):
    static_options = []
    
    res_scale = round(6 / resolution_scale)
    button_width = round(width / 6)
    button_height = round(height / 10)
    
    # Display the "save changes" button, but only have it interactable if changes were made.
    if (file_edited):
        static_options.append(draw_menu_block(img, (0 + res_scale), (0 + res_scale), ((button_width * 2) - res_scale), (button_height - res_scale), (127, 127, 127), (255, 255, 255), "Save"))
    else:
        draw_menu_block(img, (0 + res_scale), (0 + res_scale), ((button_width * 2) - res_scale), (button_height - res_scale), (63, 63, 63), (127, 127, 127), "Save")
        
    static_options.append(draw_menu_block(img, ((button_width * 2) + res_scale), (0 + res_scale), ((button_width * 4) - res_scale), (button_height - res_scale), (127, 127, 127), (255, 255, 255), "Home"))
    
    static_options.append(draw_menu_block(img, ((button_width * 4) + res_scale), (0 + res_scale), ((button_width * 6) - res_scale), (button_height - res_scale), (127, 127, 127), (255, 255, 255), "Settings"))
	
    return static_options

'''
Method that draws the dynamic menu used in the "production" menu of the system. 
Also has the logic for what pages to load and how to display the buttons in a fitting manner around the user.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the menu on.
@param eye: array is the coordinates of the eye that this button's modifiers will be based off of.
@param l_or_r: boolean tells the system if the "eye" variable given is the left or right eye, so it knows to subtract or add the modifiers.
@param eye_dist: double is the distance of pixels between the user's eyes, which is used to determine how big the joint nodes should be drawn.
@param x_mod, y_mod: double is the amount of eye distances the button should be drawn away from the eye on the respective axis.
@param item_text: string is the text that will be written inside of the button. Also identifies the button upon selection.
@return button info: array the information of the button just drawn, including its coordinates and its text, so the system can check if its being selected.
'''
def draw_dm_item(img, eye, l_or_r, eye_dist, x_mod, y_mod, item_text):  
    global resolutions
    global settings
    
    item_shift = settings["scale"]
    
    if (l_or_r):
        # This menu item must be drawn on the LEFT side of the person.
        if (y_mod >= 0):
            min_shift_x = int(round(eye[0] - ((x_mod + item_shift) * eye_dist)))
            min_shift_y = int(round(eye[1] - ((y_mod + item_shift) * eye_dist)))
            max_shift_x = int(round(eye[0] - (x_mod * eye_dist)))
            max_shift_y = int(round(eye[1] - (y_mod * eye_dist)))
            
            # The menu item must be located ABOVE the LEFT eye.
            cv2.rectangle(img, (min_shift_x, min_shift_y), (max_shift_x, max_shift_y), (40, 28, 174), -1)
            cv2.putText(img, item_text, (round(min_shift_x + ((max_shift_x - min_shift_x) / 5)), round(min_shift_y + ((max_shift_y - min_shift_y) / 3))), cv2.FONT_HERSHEY_TRIPLEX, round(resolutions[settings["resolution"]][2]), (0, 0, 0), 1, cv2.LINE_AA, False)
            
            return [item_text, min_shift_x, min_shift_y, max_shift_x, max_shift_y]
        else:
            min_shift_x = int(round(eye[0] - ((x_mod + item_shift) * eye_dist)))
            min_shift_y = int(round(eye[1] - ((y_mod + item_shift) * eye_dist)))
            max_shift_x = int(round(eye[0] - (x_mod * eye_dist)))
            max_shift_y = int(round(eye[1] - (y_mod * eye_dist)))
            
            # The menu item must be located UNDER the LEFT eye.
            cv2.rectangle(img, (min_shift_x, min_shift_y), (max_shift_x, max_shift_y), (40, 28, 174), -1)
            cv2.putText(img, item_text, (round(min_shift_x + ((max_shift_x - min_shift_x) / 4)), round(min_shift_y + ((max_shift_y - min_shift_y) / 2))), cv2.FONT_HERSHEY_TRIPLEX, round(resolutions[settings["resolution"]][2]), (0, 0, 0), 1, cv2.LINE_AA, False)
            
            return [item_text, min_shift_x, min_shift_y, max_shift_x, max_shift_y]
    else:
        # This menu item must be drawn on the RIGHT side of the person.
        if (y_mod >= 0):
            # The menu item must be located ABOVE the RIGHT eye.
            min_shift_x = int(round(eye[0] + ((x_mod + item_shift) * eye_dist)))
            min_shift_y = int(round(eye[1] - ((y_mod + item_shift) * eye_dist)))
            max_shift_x = int(round(eye[0] + (x_mod * eye_dist)))
            max_shift_y = int(round(eye[1] - (y_mod * eye_dist)))
            
            # The menu item must be located ABOVE the RIGHT eye.
            cv2.rectangle(img, (min_shift_x, min_shift_y), (max_shift_x, max_shift_y), (40, 28, 174), -1)
            cv2.putText(img, item_text, (round(min_shift_x + ((max_shift_x - min_shift_x) / 4)), round(min_shift_y + ((max_shift_y - min_shift_y) / 2))), cv2.FONT_HERSHEY_TRIPLEX, round(resolutions[settings["resolution"]][2]), (0, 0, 0), 1, cv2.LINE_AA, False)
            
            return [item_text, max_shift_x, min_shift_y, min_shift_x, max_shift_y]
        else:
            min_shift_x = int(round(eye[0] + ((x_mod + item_shift) * eye_dist)))
            min_shift_y = int(round(eye[1] - ((y_mod + item_shift) * eye_dist)))
            max_shift_x = int(round(eye[0] + (x_mod * eye_dist)))
            max_shift_y = int(round(eye[1] - (y_mod * eye_dist)))
            
            # The menu item must be located ABOVE the RIGHT eye.
            cv2.rectangle(img, (min_shift_x, min_shift_y), (max_shift_x, max_shift_y), (40, 28, 174), -1)
            cv2.putText(img, item_text, (round(min_shift_x + ((max_shift_x - min_shift_x) / 4)), round(min_shift_y + ((max_shift_y - min_shift_y) / 2))), cv2.FONT_HERSHEY_TRIPLEX, round(resolutions[settings["resolution"]][2]), (0, 0, 0), 1, cv2.LINE_AA, False)
            
            return [item_text, max_shift_x, min_shift_y, min_shift_x, max_shift_y]
'''
Method that draws all the contents of the dynamic menu of the system's production mode. This also controls what buttons should and 
 shouldn't be drawn, depending on settings and selected items on the menu, as instructed by the production_menu_behaviour function.
@author Yoran Kerbusch (24143341)

@param img: The output image the user will be shown, the one we have to draw the menu on.
@param person: arraylist is a list of X & Y arrays that tell the system where each joint is located.
@param eye_dist: double is the distance of pixels between the user's eyes, which is used to determine how big the joint nodes should be drawn.
@param x_mod: double is the amount of eye distances the menu item should be removed from the specific user.
@param init_y_mod: double is the highest amount of eye distances the buttons will start at. Negative if the buttons should start under the eyes.
@param current_menu: integer that lets the system know at which number of the dynamic menu pages this user is on.
@return buttons: array of arrays that holds the coordinates and names of each of the buttons drawn around the specific person.
'''
def draw_dynamic_menu(img, person, eye_dist, x_mod, init_y_mod, current_menu):
    item_shift = settings["scale"]
    
    l_eye = get_keypoint_by_name(person, "leye")
    r_eye = get_keypoint_by_name(person, "reye")
    cur_y_mod = init_y_mod
    
    curr = menus[current_menu]
    buttons = []
    for i in range(len(curr)):
        if ((i % 2) == 0):
            if (i != 0):
                cur_y_mod -= item_shift + 1
            # On even numbers, draw the menu item on the LEFT side of the person.
            buttons.append(draw_dm_item(img, l_eye, True, eye_dist, x_mod, cur_y_mod, curr[i]))
        else:
            # On uneven numbers, draw the menu item on the RIGHT side of the person.
            buttons.append(draw_dm_item(img, r_eye, False, eye_dist, x_mod, cur_y_mod, curr[i]))
            
    # Draw the two bottom arrow items, so the user can go between menus.
    cur_y_mod -= (item_shift + 1)
    buttons.append(draw_dm_item(img, l_eye, True, eye_dist, x_mod, cur_y_mod, "<--"))
    buttons.append(draw_dm_item(img, r_eye, False, eye_dist, x_mod, cur_y_mod, "-->"))
    
    return buttons
    
'''
Methods that control what the system should doto the script form the dynamic menu items.
@author Adam Harvey (22433643)
'''
def MakeNewFile():
    global newfileflag
    if newfileflag == False:
        newfileflag = True
        filepatchsr = os.getcwd() + "./User Scripts/" + work_file
        editFile = open(filepatchsr, "a")
        L = "#New file created for editing\n"
        editFile.writelines(L)
        editFile.close() #to change file access mode
    
    
'''
@author Adam Harvey (22433643)
'''
def MakeCopy():
    global newfilecopyflag 
    if newfilecopyflag == False:
        newfilecopyflag = True
        from shutil import copyfile
        filepatchsr = os.getcwd() + "./User Scripts/" + work_file
        try:
            copyfile(filepatchsr, FileEdit)
        except:
            pass
    
'''
@author Adam Harvey (22433643)
'''
# Passes strings to be written to the file saved as FileEdit
def WriteToFile(str):
    global file_edited
    file_edited = True
    editFile = open(FileEdit, "a")
    L = [str]
    editFile.writelines(L)
    editFile.close() #to change file access mode

'''
@author Adam Harvey (22433643)
'''
#Removes the last line from the script
def delete_last_line():
    #creates an array of all lines in the file
    readFile = open(FileEdit)
    lines = readFile.readlines()
    readFile.close()
    #removes the last item from the array and then prints out the array to the file
    rw = open(FileEdit,"a")
    rw.writelines([item for item in lines[:-1]])
    rw.close()
 
'''
@author Adam Harvey (22433643)
'''
#Copies the edited file to the target name 
def save_to_origional():
    from shutil import copyfile
    filepatchsr = os.getcwd() + "./User Scripts/" + work_file
    try:
        copyfile(FileEdit, filepatchsr)
        try:
            os.remove(FileEdit)
        except:
            pass
    except:
        pass
    
'''
@author Adam Harvey (22433643)
'''
#opens a notepad editor for use whilst editing the code
def OpenEditor():
    global editorOpen
    if editorOpen == False:
        editorOpen = True
        #notepad++ as a text editor path is hard coded an donly works with 32 bit
        cmd = [r"C:\Program Files (x86)\Notepad++\notepad++.exe", FileEdit]
        subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    
'''
@author Adam Harvey (22433643)
'''
#closes the editor and resets flags for when the editing is complete 
def CloseEditor():
    global newfilecopyflag
    global newfileflag
    global editorOpen
    if editorOpen == True:
        editorOpen = False
        newfileflag = False
        newfilecopyflag = False
        os.system("taskkill /f /im notepad++.exe")
        try:
            os.remove(FileEdit)
        except:
            pass
        

'''
Method that controls what the system should do for the production menu when one of the
 SELECTABLE buttons is selected by the user, from their own dynamic menu.
@author Adam Harvey (22433643)

@param selected_menu: string is the identifier of the button the user just selected, so we know what we must do.
@param person_number: int is the identifier number of the person that is interacting with their own dynamic menu.
@return mode: int is the menu that the system should load next.
'''
def production_behaviour(selected_button, person_number):
    global file_edited
    global ask_confirm
    global specific_menu
    
    selected = selected_button.lower()
		
    if (selected == "<--"):
        specific_menu[person_number] = ((specific_menu[person_number] - 1) + len(menus)) % len(menus)
    elif (selected == "-->"):
        specific_menu[person_number] = (specific_menu[person_number] + 1) % len(menus)
    # The functions of the production mode menu
    elif (selected == "varOne"):
        WriteToFile("varOne ")                
        pass
    elif (selected == "varTwo"):
        WriteToFile("varTwo ")
        pass
    elif (selected == "var"):
        WriteToFile("var ")
        pass
    elif (selected == "if"):
        WriteToFile("if ")
        pass
    elif (selected == "elif"):
        WriteToFile("elif ")
        pass
    elif (selected == "else"):
        WriteToFile("else ")
        pass
    elif (selected == "while"):
        WriteToFile("while ")
        pass
    elif (selected == "for"):
        WriteToFile("for ")
        pass
    elif (selected == "continue"):
        WriteToFile("continue ")
        pass
    elif (selected == "break"):
        WriteToFile("break ")
        pass
    elif (selected == "def"):
        WriteToFile("def ")
        pass
    elif (selected == "and"):
        WriteToFile("and ")
        pass
    elif (selected == "="):
        WriteToFile("= ")
        pass
    elif (selected == "=="):
        WriteToFile("== ")
        pass
    elif (selected == "!="):
        WriteToFile("!= ")
        pass
    elif (selected == ">"):
        WriteToFile("> ")
        pass
    elif (selected == "<"):
        WriteToFile("< ")
        pass
    elif (selected == "print"):
        WriteToFile("print ")
        pass
    elif (selected == "new line"):
        WriteToFile("\n")
        pass
    elif (selected == "tab"):
        WriteToFile("\t")
        pass
    elif (selected == "delete line"):
        delete_last_line()
        pass
    elif (selected == "('"):
        WriteToFile("('")
        pass
    elif (selected == "')"):
        WriteToFile("')")
        pass
    elif (selected == "'"):
        WriteToFile("'")
        pass
    elif (selected == "1"):
        WriteToFile("1 ")
        pass
    elif (selected == "2"):
        WriteToFile("2 ")
        pass
    elif (selected == "3"):
        WriteToFile("3 ")
        pass
    elif (selected == "4"):
        WriteToFile("4 ")
        pass
    elif (selected == "5"):
        WriteToFile("5 ")
        pass
    elif (selected == "6"):
        WriteToFile("6 ")
        pass
    elif (selected == "7"):
        WriteToFile("7 ")
        pass
    elif (selected == "8"):
        WriteToFile("8 ")
        pass
    elif (selected == "9"):
        WriteToFile("9 ")
        pass
    elif (selected == "0"):
        WriteToFile("0 ")
        pass
    elif (selected == "space"):
        WriteToFile(" ")
        pass
    elif (selected == "record input"):
        WriteToFile(timed_mic_input(10,1))
        pass
    elif (selected == "comma"):
        WriteToFile(", ")
        pass
    elif (selected == "full stop"):
        WriteToFile(". ")
        pass
    elif (selected == "run"):
        ##### Writes a section to the bottom of the script to wait for user to close the shell######
        WriteToFile("\nimport msvcrt as m #for the wait process\ndef wait(): #function to keep shell open for user\n\tm.getch()\nwait()")
        #####Opens the script in a shell######
        os.system(FileEdit)
        ### Removes the shell wait for button press ###
        delete_last_line()
        delete_last_line()
        delete_last_line()
        delete_last_line()
        pass
    elif (selected == "save"):
        #Save the user's changes to the file.
        save_to_origional()
        file_edited = False
    elif (selected == "home"):
        # Return to the previous menu the user was at.
        if (changes_made == True):
            # If the user made changes to the settings without saving them, ask them if they are sure they want to go back.
            ask_confirm = True
            return -1
        CloseEditor()
        
        return MAIN
    elif (selected == "settings"):
        #save the user's changes to the script before going to the settings menu.
        save_to_origional()
        file_edited = False
        CloseEditor()
        return SETT
    return -1

'''
Method that runs all the behaviour needed for the production menu, keeping the system in a
 loop here until the user selects an item to go to a different menu.
@author Yoran Kerbusch (24143341)

@param openpose: The OpenPose object that we use to record and recognise the user's body and positions.
@return MODE: int the mode to go to, but only when certain buttons are pressed, as dictated by the production_menu_behaviour() function.
'''
def production_menu(openpose):
    global resolutions
    global settings
    global new_settings
    global menus
    global specific_menu
    global work_file
    global file_edited
    global ask_confirm
    
    cap = cv2.VideoCapture(0)
    
	#Create the file for the user
    MakeNewFile()
    # A copy is made for the file and that is what is edited
    MakeCopy()
    #open the file to edit in the editor to show the user what they are doing 
    OpenEditor()
	
    while 1:
        success, img = cap.read() 
        img = cv2.resize(img, (resolutions[settings["resolution"]][0], resolutions[settings["resolution"]][1])) 
        keypoints, img = openpose.forward(img, True)
        
        # Draw the static menu bar found at the top of the screen for users.
        static_menu_bar = draw_static_prod_menu(img, resolutions[settings["resolution"]][0], resolutions[settings["resolution"]][1], resolutions[settings["resolution"]][2], file_edited)
        
        if (ask_confirm == True):
            # Do the confirmation pop-up before cheks are done on drawing people, so the background isn't cluttered.
            cap.release()
            response = confirmation_menu(openpose, img, 6, 9, "Return without saving changes to file?", "Yes", "Back to script")
            if (response == True):
                # The user does not want to save their changes made, thus return to the previous menu.
                new_settings = deepcopy(settings)
                return previous_mode
            # Otherwise, do nothing, as the user wants to continue in the settings menu.
            cap = cv2.VideoCapture(0)
            continue
        
        person_limit = 5
        if (settings["single"] == True):
            person_limit = 1
    
        for i in range(min(len(keypoints), person_limit)):
            person = keypoints[i]
            eye_dist = eye_distance(person)
            
            if (eye_dist > 0):
                # Draw the menu items around each person.
                person_menu = draw_dynamic_menu(img, person, eye_dist, settings["distance"], 0.5, specific_menu[i])
                # To allow the user to also interact with the static bar, zip the two menus together.
                interactive_menu = static_menu_bar + person_menu
                
                # Use the color hex below with the actual colour of the person, reflecting which number user they are.
                img = draw_skeleton(img, person, eye_dist, user_colors[i])
            
                selected_menu = menu_interaction(img, person, i, eye_dist, interactive_menu)
                if (selected_menu != None):
                    response = production_behaviour(selected_menu, i)
                    if (response >= 0):
                        cap.release()
                        return response
        
        cv2.imshow("output", img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            return MAIN

'''
REQUIREMENTS (MUST HAVE NVIDIA GPU):
    CUDA 9.0
    CUDNN for CUDA 9.0
    CMake
    Visual Studio 2017 C++ development tools
    Visual Studio 2017 Windows SDK x.xx.15
Place openpose folder on root directory of D:/ drive (Temporary solution)
All scripts made must be placed in openpose/Release/python/openpose directory.

@author of all adaptations: Yoran Kerbusch (24143341)
'''
if __name__ == "__main__":
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    params["num_gpu_start"] = 0
    params["disable_blending"] = False
    params["default_model_folder"] = "../../../models/"
    
    openpose = OpenPose(params)
    
    # This is the folder we'll save the user's saved Python files on.
    if not os.path.exists("./User Scripts"):
        os.makedirs("./User Scripts")
        
    # The settings menu needs to know what menu was previously opened before it.
    previous_mode = MAIN

    while 1:
        if (current_mode == MAIN):
            # We need to show the main menu, so that the other menus can be accessed. Or if the user presses "Back" it should return to the main menu.
            current_mode = main_menu(openpose)
            previous_mode = MAIN
        elif (current_mode == NEW):
            # The new program menu. This should create a file and return the path to it so production mode can open it.
            current_mode = new_menu(openpose)
            previous_mode = NEW
        elif (current_mode == LOAD):
            # The load program menu. Should shown a list of all program files in a folder and return the one the user wants to load. Or if the user presses "Back" it should return to the main menu.
            current_mode = load_menu(openpose)
            previous_mode = LOAD
        elif (current_mode == PROD):
            # We need to go to the production menu.
            current_mode = production_menu(openpose)
            previous_mode = PROD
        elif (current_mode == SETT):
            # We need to show the settings menu.
            current_mode = settings_menu(openpose, previous_mode)
            previous_mode = SETT
        else:
            # Quit the program as a whole, as we entered this.
            cv2.destroyAllWindows()
            break
# ===End of code made by Yoran Kerbusch=======================================