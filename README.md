# IP-CourseworkTwo
Code made by Yoran Kerbusch for Edge Hill University course CIS3149 Interface Programming 2018-2019 for coursework 2. Made by Yoran Kerbusch and Adam Hanvey.

This is a paired exercise made by Yoran Kerbusch and Adam Hanvey. For this exercise, students had to make a live code editor in pairs. This code editor must be able to have users create code that can be run from the software. The software must be controllable with motion, instead of keyboard & mouse. Students are free to add upon this, like adding voice control, saving, creating and loading external python files, single user modes, etc.

WARNING: To use these files, you are required to do the following:
- Download OpenPose from https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases; I RECOMMEND TO USE THE GPU VERSION (faster, more precize and less crash-prone), BUT ONLY IF YOU HAVE A POWERFUL GPU CARD IN YOUR LAPTOP/PC.
- If using the GPU version (which this project has been tested on), also make sure to install the appropriate version of NVIDIA CUDA on your computer;
- Once downloaded, go to the folder location on the disk and folder you downloaded OpenPose on "disk:/.../openpose/Release/python/openpose/";
- Drop the files from this repository into that folder;
- Use the Anaconda prompt to pip install the required extensions as written in chapter 2.1 of the documentation linked below;
- Make sure to have a 32-bit version of NotePad++ installed;
- Install the plugin "Document Manager" to NotePad++;
- Open the files with your Python editor of choice (I made & tested this code with Spyder 3.3.3);
- Click run on the file of your choice to see it run.

The final grade I received for my work on the coursework is 75.0%, and it was originally completed by me and Adam Hanvey at 8-5-2019 (8th of May, 2019).

Documentation for these files, system usage and coursework 2 (this documentation also includes the list of what work was done by me and what was done by Adam Hanvey) as a whole is found at: https://docs.google.com/document/d/1v_T05I347urVMYp1BpiPK_RG7cvEgBoC5ggGK7n3iSU/edit?usp=sharing

Features of the project include:
- Gesture recognition to select buttons, for multiple simultaneous users;
- Selecting buttons with hands, only after holding the hand over the button for a select amount of time;
- Voice recognition to select menu options instead of using hands;
- Adjustable settings (accessible both from the live code editor and main menu) for different aspects of the system;
- Python file creation, saving, existing file loading, name changing of existing files and file deletion;
- Live code editing of selected file, which can be done by several users simultaneously;
- A dynamic menu that moves with the user, staying around them and only being accessible to that user and not the others;
- Running code from the live code editor.

WARNING: If single user mode is on, then OpenPose will semi-rendomly select who that single person in its camera feed will be. It tends to prioritize the person most back in the feed (furthest away from the camera). This could not be fixed by us, as it is an underlying choice made by the OpenPose developers.
