#include "pch.h"

Neuron::Neuron()
{
	// _value and _delta are recomputed every forward/backward pass; just
	// zero everything here. Weight initialization for outgoing connections
	// happens in Network::CreateConnections (Xavier/Glorot).
	_value = 0.0;
	_bias  = 0.0;
	_delta = 0.0;
}

Neuron::~Neuron()
{
	for (Connection *p : _vecConnectionsBackward)
	{
		delete p;
	}

	_vecConnectionsBackward.clear();
	_vecConnectionsForward.clear();
}

void Neuron::Serialize(ofstream &stream)
{
	size_t connections = _vecConnectionsBackward.size();

	stream.write((const char *)&_bias, sizeof(double));
	stream.write((const char *)&connections, sizeof(size_t));

	for (Connection *p : _vecConnectionsBackward)
	{
		p->Serialize(stream);
	}
}

void Neuron::DeSerialize(ifstream &stream, Layer *parentLayer)
{
	size_t connections = 0;

	stream.read((char *)&_bias, sizeof(double));
	stream.read((char *)&connections, sizeof(size_t));

	//wprintf(L"        read %zd connections\n", connections);
	for (size_t i = 0; i < connections; i++)
	{
		Connection *p = new Connection(0.0);
		p->DeSerialize(stream);
		p->_child = this;
		p->_parent = parentLayer->_vecNeurons[i];

		_vecConnectionsBackward.push_back(p);
		p->_parent->_vecConnectionsForward.push_back(p);
	}
}

void Neuron::AddConnection(Neuron *pChild)
{
	Connection *p = new Connection();
	p->_parent = this;
	p->_child = pChild;

	_vecConnectionsForward.push_back(p);
	pChild->_vecConnectionsBackward.push_back(p);
}

void Neuron::SetValueFromSample(imagesample *pSample, int index)
{
	//_value = pSample->pixels[index] > 0 ? 1.0 : 0.0;
	_value = Utils::Relu(pSample->pixels[index], 255); 
	//_value = Utils::Sigmoid(pSample->pixels[index]);
}



