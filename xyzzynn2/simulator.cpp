#include "pch.h"
#include <random>
#include <algorithm>

static double _maxaccuracy;
static int _sample;

void Simulator::Initialize()
{
	_imagesTraining.Initialize(L"train-images-idx3-ubyte", 16);
	_labelsTraining.Initialize(L"train-labels-idx1-ubyte", 8);

	_imagesTest.Initialize(L"t10k-images-idx3-ubyte", 16);
	_labelsTest.Initialize(L"t10k-labels-idx1-ubyte", 8);

}

void Simulator::DumpData()
{
	for (UINT i = 0; i < _imagesTest.Items(); i++)
	{
		imagesample *pis = _imagesTest.GetItem(i);

		wprintf(L"%d: label=%d\n", i, *(uint8_t*)_labelsTest.GetItem(i));

		DumpSample(pis);

	}
	wprintf(L"images count: %d, labels count: %d", _imagesTest.Items(), _labelsTest.Items());
}

void Simulator::DumpSample(imagesample *pis)
{
	int ipos = 0;
	for (UINT y = 0; y < 28; y++)
	{
		for (UINT x = 0; x < 28; x++)
		{
			if (pis->pixels[x + (y * 28)] > 0)
			{
				wprintf(L"$");
			}
			else
			{
				wprintf(L" ");
			}
		}

		wprintf(L"\n");
	}
	wprintf(L"--------------------------------------------\n");
}

void Simulator::DumpNetwork()
{
	if (!_pNetwork)
	{
		return;
	}

	for (size_t l = 0; l < _pNetwork->_layers.size(); ++l)
	{
		const LayerDesc& d = _pNetwork->_layers[l];
		wprintf(L"layer %zu: size=%u, prev=%u, weights=%u\n",
		        l, d.neuronCount, d.prevNeuronCount,
		        d.neuronCount * d.prevNeuronCount);

		const float* val = _pNetwork->_values.data() + d.neuronOffset;
		for (uint32_t j = 0; j < d.neuronCount; ++j)
		{
			wprintf(L"\t neuron %u value=%f\n", j, val[j]);

			if (d.prevNeuronCount > 0)
			{
				const float* W = _pNetwork->_weights.data()
				                 + d.weightOffset + (size_t)j * d.prevNeuronCount;
				for (uint32_t i = 0; i < d.prevNeuronCount; ++i)
				{
					wprintf(L"\t\t connection: weight = %f\n", W[i]);
				}
			}
		}
	}
}

bool Simulator::LoadNetwork(const wchar_t *wz)
{
	if (!_pNetwork)
	{
		_pNetwork = new Network();
	}

	return _pNetwork->LoadNetwork(wz);
}

void Simulator::SaveNetwork(const wchar_t *wz)
{
	if (_pNetwork)
	{
		_pNetwork->SaveNetwork(wz);
	}
}

void Simulator::CreateNetwork()
{
	_pNetwork = new Network();
	_pNetwork->AddLayer(784); // input layer
	_pNetwork->AddLayer(256); // hidden layer
	_pNetwork->AddLayer(10);  // output layer

	_pNetwork->CreateConnections();

	//DumpNetwork();
}

double Simulator::CalculateTotalCost()
{
	// run through all samples
	double totalcost = 0;
	for (UINT32 i = 0; i < _imagesTraining.Items(); i++)
	{
		imagesample *pSample = _imagesTraining.GetItem(i);
		uint8_t label = *(uint8_t*)(_labelsTraining.GetItem(i));

		double cost = _pNetwork->CalculateCost(pSample, label);
		totalcost += cost;

		if (i % 1000 == 0)
		{
			wprintf(L"processing sample=%d/%d\n", i, _imagesTraining.Items());
		}
	}

	return totalcost;
}

void Simulator::Learn(double rate, int32_t epochs, int32_t trainLimit)
{
	//DumpNetwork();

	wprintf(L"learn parameters: rate=%f, epochs=%d\n", rate, epochs);
	std::mt19937 rng(std::random_device{}()); // Seed the generator

	for (int32_t epoch = 0; epoch < epochs; epoch++)
	{
		// run through all samples
		double totalcost = 0;
		int samples = 0;

		if (trainLimit == 0)
		{
			trainLimit = _imagesTraining.Items();
		}

		vector<int> vecOrder;
		uint32_t iTraining = min(_imagesTraining.Items(), (uint32_t)trainLimit);
		vecOrder.reserve(_imagesTraining.Items());

		for (uint32_t i = 0; i < _imagesTraining.Items(); i++)
		{
			vecOrder.push_back(i);
		}

		// Use the per-Learn() rng seeded outside this loop so each epoch
		// gets a different permutation.
		std::shuffle(std::begin(vecOrder), std::end(vecOrder), rng);

		uint32_t timeStarted = GetTickCount();

		for (UINT32 i = 0; i < iTraining; i++)
		{
			imagesample *pSample = _imagesTraining.GetItem(vecOrder[i]);
			uint8_t label = *(uint8_t*)(_labelsTraining.GetItem(vecOrder[i]));

			if (i > 0 && (i % 1000) == 0)
			{
				wprintf(L".");
			}

			samples++;

			double cost = _pNetwork->BatchForward(pSample, label);
			totalcost += cost;

			_pNetwork->PropagateBackward(label, rate);
			
			if (i == 29)
			{
				uint32_t sec = ((GetTickCount() - timeStarted) / 30);
				wprintf(L"Sample: %d ms, estimated epoch time: %f minutes\r\n", 
					sec, 
					(sec * iTraining) / 60000.0f);
			}
		}

		wprintf(L"\r\nepoch %d: trained on %d samples\r\n", epoch, samples);
		wprintf(L"epoch: %d - rate: %f - total cost: %f\n", epoch, rate, totalcost);

		AccuracyTest(true, false);	
	}
}

void Simulator::AccuracyTest(bool fSaveMax, bool fDumpWrong)
{
	double correct = 0;
	double wrong = 0;
	double accuracy = 0;

#define USE_TEST_SET 1
#if USE_TEST_SET
	for (UINT32 i = 0; i < _imagesTest.Items(); i++)
	{
		imagesample *pSample = _imagesTest.GetItem(i);
		uint8_t label = *(uint8_t*)(_labelsTest.GetItem(i));
#else
	for (UINT32 i = 0; i < _imagesTraining.Items(); i++)
	{
		//if (i > 500)
		//	continue;

		imagesample *pSample = _imagesTraining.GetItem(i);
		uint8_t label = *(uint8_t*)(_imagesTraining.GetItem(i));
#endif

	//	if (label != 0 && label != 1)
	//	    continue;

		uint8_t bGuess = 0;

		if (_pNetwork->AccuracyTest(pSample, label, &bGuess))
		{
			correct++;
		}
		else
		{
			wrong++;

			if (fDumpWrong)
			{
				wprintf(L"label: %d, guess: %d\n", label, bGuess);
				DumpSample(pSample);
			}
		}
	}

	accuracy = correct / (correct + wrong);
	if (accuracy > _maxaccuracy)
	{
		_maxaccuracy = accuracy;

		if (fSaveMax)
		{
			SaveNetwork(L"maximum.xnn");
		}
	}

	wprintf(L"correct=%f, wrong=%f, accuracy = %f (max=%f)\n", correct, wrong, accuracy, _maxaccuracy);

}
