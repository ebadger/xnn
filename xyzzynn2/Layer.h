#pragma once
#include "pch.h"

class Neuron;

class Layer
{
public:
	Layer(double multiplier);
	~Layer();

	void LoadInputLayer(imagesample *pSample);

	bool CreateNeurons(int iNeurons);
	void Serialize(ofstream &stream);
	void DeSerialize(ifstream &stream, Layer *pParent);

	vector<Neuron *> _vecNeurons;
	double m_multiplier = 0.0;
	double _totalNeuronScore = 0.0;
};