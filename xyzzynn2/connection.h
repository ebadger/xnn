#pragma once
#include "pch.h"

class Neuron;

class Connection
{
public:
	Connection(double weight);
	Connection();
	~Connection();

	void Serialize(ofstream &stream);
	void DeSerialize(ifstream &stream);

	Neuron *_parent		= nullptr;
	Neuron *_child		= nullptr;
	double _weight		= 0.0;
	double _originalWeight = 0.0;

};