#include "pch.h"

bool Network::AddLayer(int iNeurons, double multiplier)
{
	if (_fInitialized)
	{
		return false;
	}

	Layer *pLayer = new Layer(multiplier);
	pLayer->CreateNeurons(iNeurons);
	
	_vecLayers.push_back(pLayer);

	return true;
}

bool Network::CreateConnections()
{
	if (_fInitialized)
	{
		return false;
	}

	// each neuron in the layer connects to each neuron in layer + 1
	// and back

	for (uint32_t i = 0; i < _vecLayers.size() - 1; i++)
	{
		Layer *parent = _vecLayers[i];
		Layer *child = _vecLayers[i + 1];
        double limit  = std::sqrt(6.0 / (parent->_vecNeurons.size() + child->_vecNeurons.size()));

		for (Neuron *pParentNeuron : parent->_vecNeurons)
		{
			for (Neuron *pChildNeuron : child->_vecNeurons)
			{
				pParentNeuron->AddConnection(pChildNeuron);
				pParentNeuron->_vecConnectionsForward.back()->_weight =
                Utils::RandomDouble(-limit, limit);
			}
		}
	}

	return true;
}


double Network::CalculateCost(imagesample *pSample, uint8_t label)
{
	double cost = 0.0;

	Layer *pInputLayer = _vecLayers[0];
	int i = 0;

	for (Neuron *pNeuron : pInputLayer->_vecNeurons)
	{
		CheckConditionFailFast(i < 784);
		// set initial activation value
		pNeuron->_value = pSample->pixels[i++];
	}

	for (UINT l = 1; l < _vecLayers.size(); l++)
	{
		Layer *pLayer = _vecLayers[l];

		for (Neuron *pNeuron : pLayer->_vecNeurons)
		{
			pNeuron->_value = 0.0;

			for (Connection *pConnection : pNeuron->_vecConnectionsBackward)
			{
				pNeuron->_value += (pConnection->_parent->_value) * (pConnection->_weight);
			}

			pNeuron->_value += pNeuron->_bias;
			pNeuron->_value = Utils::Sigmoid(pNeuron->_value);
			//wprintf(L"value=%f\r\n", pNeuron->_value);
		}
	}
	
	// now look at the output layer

	Layer *pOutputLayer = _vecLayers[_vecLayers.size() - 1];
	int j = 0;
	for (Neuron *pNeuron : pOutputLayer->_vecNeurons)
	{
		CheckConditionFailFast(j < 0x256);
		double expected = 0.0;
		double actual = 0.0;
		if (label == (uint8_t)j)
		{
			expected = 1.0;
		}

		actual = expected - pNeuron->_value;
		actual = actual * actual;
		cost += actual;
		//wprintf(L"%d: %f, %f\r\n", j++, pNeuron->_value, actual);

		j++;
	}

	return cost;
}


void Network::Clear()
{
	for (Layer *p : _vecLayers)
	{
		delete p;
	}
	_vecLayers.clear();
}

void Network::OutputNetworkInfo()
{
	wprintf(L"Layers: %d [", (uint32_t)_vecLayers.size());
	
	for (Layer* l : _vecLayers)
	{
		wprintf(L" %d ", (uint32_t)l->_vecNeurons.size());
	}

	wprintf(L"]\r\n");
}

bool Network::LoadNetwork(const wchar_t *wzFileName)
{
	Clear();

	ifstream istream;
	istream.open(wzFileName, ios::in | ios::binary);

	if (istream.is_open())
	{
		// read the # of layers
		size_t layers = 0;
		istream.read((char *)&layers, sizeof(size_t));
		
		wprintf(L"read %zd layers\n", layers);

		for (size_t i = 0; i < layers; i++)
		{
			Layer *pLayer = new Layer(0.0);
			_vecLayers.push_back(pLayer);

			Layer *pParent = nullptr;

			if (i > 0)
			{
				pParent = _vecLayers[i - 1];
			}

			pLayer->DeSerialize(istream, pParent);
		}

		wprintf(L"loaded: %s\r\n", wzFileName);
		OutputNetworkInfo();

		return true;
	}
	else
	{
		WCHAR cdir[MAX_PATH];
		GetCurrentDirectoryW(MAX_PATH, cdir);
		wprintf(L"Failed to open %s in dir: %s\r\n", wzFileName, cdir);
		return false;
	}
}

void Network::SaveNetwork(const wchar_t *wzFileName)
{
	std::ofstream ostream;
	ostream.open(wzFileName, ios::binary);
	
	if (ostream.is_open())
	{
		Serialize(ostream);
		wprintf(L"saved: %s\n", wzFileName);
	}

	ostream.close();

}

void Network::Serialize(ofstream &stream)
{
	size_t layers = _vecLayers.size();
	stream.write((const char *)&layers, sizeof(size_t));

	for (Layer *p : _vecLayers)
	{
		p->Serialize(stream);
	}
}

void Network::PropagateForward(imagesample *pSample)
{
	// start with the first hidden layer
	for (UINT l = 1; l < _vecLayers.size(); l++)
	{
		Layer *pLayer = _vecLayers[l];

		for (Neuron *pNeuron : pLayer->_vecNeurons)
		{
			//double preval = pNeuron->_value;
			double v = 0.0;

			// look back to the previous layer to calculate neuron scores at this layer
			for (Connection *pConnection : pNeuron->_vecConnectionsBackward)
			{
				//double pw = pConnection->_parent->_value;
				//v += (pw * (1 - pw)) * pConnection->_weight;
				v += pConnection->_weight * pConnection->_parent->_value;
			}

			pNeuron->_value = v;

			pNeuron->_value += pNeuron->_bias;
			//double sigval = Utils::Sigmoid(pNeuron->_value);
			//double sigval = Utils::Relu(pNeuron->_value, (double)pNeuron->_vecConnectionsBackward.size());
			double sigval = Utils::Sigmoid(pNeuron->_value);
			//wprintf(L"neuron=%f, sig=%f\n", pNeuron->_value, sigval);
			pNeuron->_value = sigval;
		}
	}
}

double Network::BatchForward(imagesample* pSample, uint8_t label)
{
	Layer* pInputLayer = _vecLayers[0];

	pInputLayer->LoadInputLayer(pSample);

	PropagateForward(pSample);

	// now look at the output layer

	Layer* pOutputLayer = _vecLayers[_vecLayers.size() - 1];
	int digit = 0;
	double totalcost = 0.0;
	double expected = 0.0;

	//wprintf(L"%d,", label);

	for (Neuron* pNeuron : pOutputLayer->_vecNeurons)
	{
		double cost = 0.0;
		expected = 0;
		double out = pNeuron->_value;


		if (label == (uint8_t)digit)
		{
			expected = 1.0;
		}

		cost = expected - out;

		cost = (cost * cost);

		totalcost += cost;
		digit++;
	}

	//wprintf(L"\n");

	//wprintf(L"totalcost %f\n", totalcost);

	return totalcost;
}


bool Network::AccuracyTest(imagesample *pSample, uint8_t label, uint8_t*pbGuess)
{
	double cost = 0.0;
	*pbGuess = 0xFF;
	Layer *pInputLayer = _vecLayers[0];
	pInputLayer->LoadInputLayer(pSample);

	PropagateForward(pSample);

	// now look at the output layer

	Layer *pOutputLayer = _vecLayers[_vecLayers.size() - 1];
	int j = 0;
	int maxj = 0;
	double maxvalue = 0.0;

	for (Neuron *pNeuron : pOutputLayer->_vecNeurons)
	{
		CheckConditionFailFast(j < 0x256);

		if (pNeuron->_value > maxvalue)
		{
			maxj = j;
			maxvalue = pNeuron->_value;
		}

		j++;
	}


	*pbGuess = maxj;

	if (label == (uint8_t)maxj)
	{
		return true;
	}
	else
	{
		return false;
	}

}

void Network::PropagateBackward(uint8_t label, double rate)
{
    // 1) δ at the output layer
    Layer* pOut = _vecLayers.back();
    for (size_t k = 0; k < pOut->_vecNeurons.size(); ++k)
    {
        Neuron* n = pOut->_vecNeurons[k];
        double y = (label == (uint8_t)k) ? 1.0 : 0.0;
        double dLda = n->_value - y;          // d(½(a-y)²)/da
        n->_delta  = dLda * n->_value * (1.0 - n->_value); // σ'(z) for sigmoid
    }

    // 2) δ for every hidden layer, back to (but not including) the input
    for (int l = (int)_vecLayers.size() - 2; l >= 1; --l)
    {
        for (Neuron* j : _vecLayers[l]->_vecNeurons)
        {
            double sum = 0.0;
            for (Connection* p : j->_vecConnectionsForward)   // children
            {
                sum += p->_weight * p->_child->_delta;        // Σ w_kj · δ_k
            }
            j->_delta = sum * j->_value * (1.0 - j->_value);  // σ'(z_j)
        }
    }

    // 3) Apply weight + bias updates (now that all δ are known)
    for (size_t l = 1; l < _vecLayers.size(); ++l)
    {
        for (Neuron* j : _vecLayers[l]->_vecNeurons)
        {
            for (Connection* p : j->_vecConnectionsBackward)  // parents
            {
                p->_weight -= rate * j->_delta * p->_parent->_value;
            }
            j->_bias -= rate * j->_delta;
        }
    }
}