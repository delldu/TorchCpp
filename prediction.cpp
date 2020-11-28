/************************************************************************************
***
*** Copyright 2019 Dell(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, 2019-09-28 00:37:40
***
************************************************************************************/


// One-stop header.
#include <torch/script.h>

// headers for opencv
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>


#define IMAGE_SIZE 224
#define CHANNELS 3
#define TOP_K 3

bool load_image(std::string file_name, cv::Mat & image)
{
	image = cv::imread(file_name);	// CV_8UC3
	if (image.empty() || !image.data) {
		return false;
	}
	cv::cvtColor(image, image, CV_BGR2RGB);

	// scale image to fit
	cv::Size scale(IMAGE_SIZE, IMAGE_SIZE);
	cv::resize(image, image, scale);

	// convert [unsigned int] to [float]
	image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

	return true;
}

bool load_label(std::string file_name, std::vector < std::string > &labels)
{
	std::ifstream ifs(file_name);
	if (!ifs) {
		return false;
	}
	std::string line;
	while (std::getline(ifs, line)) {
		labels.push_back(line);
	}
	return true;
}

int main(int argc, const char *argv[])
{
	if (argc != 4) {
		std::cerr << "Usage: " << argv[0] << " model lable image" << std::endl;
		return -1;
	}

	torch::jit::script::Module module;
	try {
		module = torch::jit::load(argv[1]);
	}
	catch(const c10::Error & e) {
		std::cerr << "Loading model error." << std::endl;
		return -1;
	}

	// to GPU
	module.to(at::kCUDA);
	// torch::Device device(torch::kCPU);
	// if (torch::cuda::is_available()) {
	// 	std::cout << "CUDA is available! Training on GPU." << std::endl;
	// 	device = torch::Device(torch::kCUDA);
	// }

	std::vector < std::string > labels;
	if (!load_label(argv[2], labels)) {
		std::cerr << "Loading label error." << std::endl;
		return -1;
	}

	cv::Mat image;

	if (!load_image(argv[3], image)) {
		std::cerr << "Loading image error." << std::endl;
	}

	auto input_tensor = torch::from_blob(image.data, { 1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS });
	input_tensor = input_tensor.permute( {0, 3, 1, 2} );
	input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
	input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
	input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

	// to GPU
	input_tensor = input_tensor.to(at::kCUDA);

	// Test speed ...
	for (int i = 0; i < 1000; i++) {
		if (i % 100 == 0) {
			std::cout << i << " ... " << std::endl;
		}
		module.forward( {input_tensor} ).toTensor();
	}

	torch::Tensor out_tensor;
	out_tensor = module.forward( {input_tensor} ).toTensor();

	auto results = out_tensor.sort(-1, true);
	auto softmaxs = std::get < 0 > (results)[0].softmax(0);
	auto indexs = std::get < 1 > (results)[0];

	std::cout << "Class Result:" << std::endl;
	for (int i = 0; i < TOP_K; ++i) {
		auto idx = indexs[i].item < int >();
		std::cout << "  " << labels[idx];
		std::cout << ": " << softmaxs[i].item < float >() * 100.0f << "%" << std::endl;
	}
}
