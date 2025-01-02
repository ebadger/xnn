#pragma once
#include "pch.h"

class Utils
{
public:
	static uint32_t LittleToBigEndian(uint32_t num);
	static double RandomDouble(double min, double max);
	static double Sigmoid(double x);
	static double SigmoidDerivative(double x);
	static double Relu(double x, double max);

};
