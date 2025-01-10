#include "pch.h"
#include <random>

Neuron::Neuron()
{
	std::random_device rd; // Obtain a random number from hardware std::mt19937 gen(rd())
	std::mt19937 gen(rd()); // Seed the generator
	
	std::uniform_real_distribution<> dis(.0, 1.0); // Range: [0.0, 1.0] // Generate the random number and assign it to variable 'x' double x = dis(gen);
	_value = dis(gen);
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
	p->_originalWeight = p->_weight;

	_vecConnectionsForward.push_back(p);
	pChild->_vecConnectionsBackward.push_back(p);
}

void Neuron::SetValueFromSample(imagesample *pSample, int index)
{
	//_value = pSample->pixels[index] > 0 ? 1.0 : 0.0;
	_value = Utils::Relu(pSample->pixels[index], 255); 
	//_value = Utils::Sigmoid(pSample->pixels[index]);
}


void Neuron::BackPropagateError(double din, int layer, double rate)
{
	double d = _value * (1.0 - _value) * din * Utils::SigmoidDerivative(_value);
		
	for (Connection* p : _vecConnectionsBackward)
	{	
		p->_parent->BackPropagateError(d * p->_weight, layer + 1, rate);
		p->_weight = (rate * din * p->_parent->_value) + p->_weight;
	}
}

#if 0

void Neuron::BackPropagateError(double error, int layer, double rate)
{
	for (Connection* p : _vecConnectionsBackward)
	{
		//double adjust =  -1.0 * rate * error * Utils::ReluDerivative(p->_parent->_value + DBL_EPSILON, (double)p->_parent->_vecConnectionsBackward.size());
	
		// todo: instead of making the adjustment proportional to the number of neurons in the layer
		// sum the neuron values and calculate the parent neuron's proportion to the other values
		// use that to adjust proportional to not only the number of neurons, but that neuron's relative
		// weight

		double errorWeight = 100.0 * p->_parent->_value * p->_parent->_pLayer->_totalNeuronScore;

		// take the error, and normalize it based on the number of neurons in this layer
		// make the error proportional to that number
		double propError = (error * errorWeight);

		if (!isnan(p->_weight + propError * rate))
		{
			// TODO: this weight update needs to happen at the end, otherwise the weights are changing
			// alternatively and better, the average could be calculated at each layer and then used during
			// back prop

			p->_weight += propError * rate;
		}

		p->_parent->BackPropagateError(error, layer + 1, rate);
	}
}

void Neuron::BackPropagateError(double error, int layer, double rate)
{
	for (Connection* p : _vecConnectionsBackward)
	{
		//double adjust =  -1.0 * rate * error * Utils::ReluDerivative(p->_parent->_value + DBL_EPSILON, (double)p->_parent->_vecConnectionsBackward.size());
		double adjust = error * (p->_parent->_value + DBL_EPSILON) * (double)(p->_parent->_vecConnectionsBackward.size() + 1.0);

		if (!isnan(p->_weight + adjust * rate))
		{
			p->_weight += adjust * rate;
		}

		p->_parent->BackPropagateError(adjust, layer + 1, rate);
	}
}

void Neuron::BackPropagateError(double error, int layer, double rate)
{
	// Compute local gradient for output neurons or hidden neurons
	double localGradient;
	if (layer == 0) // Assuming 0 is the output layer
	{
		localGradient = error; // For output layer, error is directly used
	}
	else
	{
		localGradient = 2.0 * error * Utils::SigmoidDerivative(_value); // For hidden layers
	}

	// First, calculate errors for all parent neurons without updating weights
	std::vector<std::pair<Neuron*, double>> errorsToPropagate(_vecConnectionsBackward.size());
	int i = 0;

	for (Connection* p : _vecConnectionsBackward)
	{
		double sumError = 0.0;
		for (Connection* conn : p->_parent->_vecConnectionsForward)
		{
			sumError += conn->_weight * localGradient;
		}
		
		errorsToPropagate[i++] = std::make_pair(p->_parent, sumError);

		double deltaWeight = rate * localGradient * p->_parent->_value;
		if (!isnan(p->_weight + deltaWeight))
		{
			p->_weight += deltaWeight;
		}
	}

	// Finally, propagate errors
	for (auto& pair : errorsToPropagate)
	{
		pair.first->BackPropagateError(pair.second, layer + 1, rate);
	}
}

#endif