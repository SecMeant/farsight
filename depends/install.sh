git clone https://github.com/libusb/libusb
cd libusb
git checkout 51b10191033ca3a3819dcf46e1da2465b99497c2
git apply ../libusb_buffering.patch


./autogen.sh
./bootstrap.sh
./configure --prefix=/usr/
sudo make install
