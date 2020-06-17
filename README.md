# farsight

to install patched libusb run depends/install.sh

to prepare environment for running locally source depends/init
for running with X forwarding source depends/remote_init

to properly configure usbcore run depends/fixusbcore.sh with proper privileges

# Interface
## Opencv
 b - set base image for choosen camera. Should be done at first allways. \
 l - load camera config. \
 s - caputer postion found by aruco \
 c - load chessboard photos to performe live camera configuration \
 n - find nearest point of meassured object \
 r - meassure reference object \
 x - rerender scene to opengl \
 1 - select first camera \
 2 - select second camera \
 trackbar - you can use it to change floor level of current scene \

## OpenGL


# Configuration
1. First of all configure base image, you should assure that current scene is prepared and when it comes to meassurment the only new thing in current scene will be wanted object
2. Once you capture your base images, you should configure your cameras. Load camaera config if you have one or get chessboard images from specified folder pressing 'c'
3. Set you object on preconfigured scene and get nearest points pressing 'n'
4. Set correct floor level, remeber that visible value is deivided by 1000, so maximum floor level value is 1.0
5. Finaly meassure object
6. You can press 'x' to reload config to opengl
