#pragma once
#include "pch.h"

class Neuron
{
public:
	Neuron();
	~Neuron();

	void AddConnection(Neuron *);
	void BackPropagateError(double error, int layer, double rate);
	void SetValueFromSample(imagesample *pSample, int index);

	void Serialize(ofstream &stream);
	void DeSerialize(ifstream &stream, Layer *parentLayer);

	double _bias							= 0.0;
	double _value							= 0.0;

	Layer *_pLayer							= nullptr;

	vector<Connection *> _vecConnectionsForward;
	vector<Connection *> _vecConnectionsBackward;

	vector<double> _vecCostBatch;
};
