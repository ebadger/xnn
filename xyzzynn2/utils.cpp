#include <pch.h>
#include <random>

uint32_t Utils::LittleToBigEndian(uint32_t num)
{
	int byte0, byte1, byte2, byte3;
	byte0 = (num & 0x000000FF) >> 0;
	byte1 = (num & 0x0000FF00) >> 8;
	byte2 = (num & 0x00FF0000) >> 16;
	byte3 = (num & 0xFF000000) >> 24;
	return((byte0 << 24) | (byte1 << 16) | (byte2 << 8) | (byte3 << 0));
}

double Utils::RandomDouble(double min, double max)
{
	std::random_device rd; // Obtain a random number from hardware std::mt19937 gen(rd())
	std::mt19937 gen(rd()); // Seed the generator

	std::uniform_real_distribution<> dis(min, max); // Range: [1.0, 10.0] // Generate the random number and assign it to variable 'x' double x = dis(gen);
	return dis(gen);
}

double Utils::Sigmoid(double x)
{
	double exp_value;
	double return_value;

	/*** Exponential calculation ***/
	exp_value = exp((double)-x);

	/*** Final sigmoid value ***/
	return_value = 1 / (1 + exp_value);

	return return_value;
}

double Utils::SigmoidDerivative(double x)
{
	double sigmoidValue = Sigmoid(x);
	return sigmoidValue * (1 - sigmoidValue);
}

double Utils::Relu(double x, double max)
{
	double r = x / max;
	return r < 0 ? 0 : r;
}

double Utils::ReluDerivative(double x, double max) 
{
	double r = x / max;
	return r < 0 ? 0 : (1.0 / max);
}
