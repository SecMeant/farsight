#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/logger.h>

#include <fmt/format.h>
#include <memory>
#include <atomic>

#include "viewer.h"

std::atomic_flag continue_flag;

extern "C" {
#include <signal.h>
#include <unistd.h>
}

void sigint_handler(int signo)
{
	fmt::print("Signal handler\n");

	if (signo == SIGINT)
	{
		fmt::print("Got SIGINT\n");
		continue_flag.clear();
	}
}

int main()
{
	continue_flag.test_and_set();

	if (signal(SIGINT, sigint_handler) == SIG_ERR)
	{
		fmt::print("Failed to register signal handler.\n");
		return -2;
	}

	libfreenect2::Freenect2 freenect2;

	if (freenect2.enumerateDevices() == 0)
	{
		fmt::print("No devices connected\n");
		return -1;
	}

	std::string serial = freenect2.getDefaultDeviceSerialNumber();

	fmt::print("Connecting to the device with serial: {}\n", serial);

	auto pipeline	= new libfreenect2::OpenCLPacketPipeline;
	auto dev	= freenect2.openDevice(serial, pipeline);

	libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color);
	libfreenect2::FrameMap frames;

	dev->setColorFrameListener(&listener);

	if (!dev->startStreams(true, false))
		return -1;
	
	fmt::print(
		"Connecting to the device\n"
		"Device serial number	: {}\n"
		"Device firmware	: {}\n"
		, dev->getSerialNumber()
		, dev->getFirmwareVersion()
	);

	Viewer viewer;
	viewer.initialize();

	while (continue_flag.test_and_set())
	{
		if (!listener.waitForNewFrame(frames, 10*1000))
		{
			fmt::print("TIMEDOUT !\n");
			return -1;
		}

		libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];

		viewer.addFrame("RGB", rgb);
		viewer.render();

		listener.release(frames);
	}

	dev->stop();
	dev->close();

	return 0;
}

