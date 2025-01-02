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
	for (Layer *pLayer : _pNetwork->_vecLayers)
	{
		wprintf(L"layer: size=%zd\n", pLayer->_vecNeurons.size());
		for (Neuron *pNeuron : pLayer->_vecNeurons)
		{
			wprintf(L"\t neuron value=%f: fw=%zd, bw=%zd\n", pNeuron->_value, pNeuron->_vecConnectionsForward.size(), pNeuron->_vecConnectionsBackward.size());
			for (Connection *pConnection : pNeuron->_vecConnectionsForward)
			{
				wprintf(L"\t\t connection: weight = %f\n", pConnection->_weight);
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
	_pNetwork->AddLayer(784, 0); // input layer
	//_pNetwork->AddLayer(16, 0);  // hidden layer
	_pNetwork->AddLayer(100, 0);  // hidden layer
	_pNetwork->AddLayer(10, 0);  // output layer

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

void Simulator::Learn(double rate, int32_t epochs)
{
	//DumpNetwork();

	wprintf(L"learn parameters: rate=%f, epochs=%d\n", rate, epochs);

	for (int32_t epoch = 0; epoch < epochs; epoch++)
	{
		// run through all samples
		double totalcost = 0;
		int samples = 0;

		vector<int> vecOrder;
		uint32_t iTraining = min(_imagesTraining.Items(), 10000);

		for (uint32_t i = 0; i < iTraining; i++)
		{
			vecOrder.push_back(i);
		}

		auto rng = std::default_random_engine{};
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

			_pNetwork->BatchBackward(rate);
			
			if (i == 99)
			{
				uint32_t sec = ((GetTickCount() - timeStarted) / 100);
				wprintf(L"Sample: %d ms, estimated epoch time: %d minutes\r\n", 
					sec, 
					(sec * iTraining) / 60000);
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
