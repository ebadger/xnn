#include "pch.h"

Layer::Layer(double multiplier)
{
	m_multiplier = multiplier;
}

Layer::~Layer()
{
	for (Neuron *p : _vecNeurons)
	{
		delete p;
	}

	_vecNeurons.clear();
}

void Layer::Serialize(ofstream &stream)
{
	size_t neurons = _vecNeurons.size();
	stream.write((const char *)&neurons, sizeof(size_t));

	for (Neuron *p : _vecNeurons)
	{
		p->Serialize(stream);
	}
}

void Layer::DeSerialize(ifstream &stream, Layer *pParent)
{
	size_t neurons = 0;

	stream.read((char *)&neurons, sizeof(size_t));
	//wprintf(L"   read %zd neurons\n", neurons);

	for (int i = 0; i < neurons; i++)
	{
		Neuron *p = new Neuron();
		p->_pLayer = this;
		_vecNeurons.push_back(p);
		p->DeSerialize(stream, pParent);
	}
}

bool Layer::CreateNeurons(int iNeurons)
{
	for (int i = 0; i < iNeurons; i++)
	{
		Neuron *pNeuron = new Neuron();
		pNeuron->_pLayer = this;
		_vecNeurons.push_back(pNeuron);
	}

	return true;
}

void Layer::LoadInputLayer(imagesample *pSample)
{
	int i = 0;
	for (Neuron *pNeuron : _vecNeurons)
	{
		CheckConditionFailFast(i < 784);
		// set initial activation value
		pNeuron->SetValueFromSample(pSample, i++);

#if 0
		if (pNeuron->_value > 0.5)
		{
			wprintf(L"$");
		}
		else
		{
			wprintf(L" ");
		}

		if (i % 28 == 0)
		{
			wprintf(L"\n");
		}
#endif

	}

}