ROOT_DIR="$(pwd)"

have_prog() {
    [ -x "$(which $1)" ]
}

install_apt(){
    sudo apt install libusb-1.0-0-dev
    sudo apt install binutills-gold
    sudo apt install freeglut3
    sudo apt install freeglut3-dev
    sudo apt install libglew-dev
    sudo apt install mesa-common-dev
    sudo apt install build-essentials
    sudo apt install libglew1.5-dev libglm-dev
}

if have_prog apt ; then 
    install_apt
else
    echo 'No package manager found!'
    exit 2
fi

git clone https://github.com/libjpeg-turbo/libjpeg-turbo
cd libjpeg-turbo
mkdir build
cd build
cmake ..
sudo make install -j"$(nproc)"

cd "$ROOT_DIR"

git clone https://github.com/glfw/glfw
cd glfw
mkdir build
cd build
cmake ..
sudo make install -j"$(nproc)"

cd "$ROOT_DIR"

git clone https://github.com/OpenKinect/libfreenect2
cd libfreenect2
sed -i "s/OPENGL_gl_LIBRARY/OPENGL_LIBRARIES/" CMakeLists.txt
mkdir build
cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install -j"$(nproc)"

cd "$ROOT_DIR"

git clone https://github.com/libusb/libusb
cd libusb
git checkout 51b10191033ca3a3819dcf46e1da2465b99497c2
git apply ../libusb_buffering.patch


./autogen.sh
./bootstrap.sh
./configure --prefix=/usr/
sudo make install
