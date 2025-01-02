#pragma once
#include "pch.h"

class Network
{
public:
	bool AddLayer(int uiNeurons, double multiplier);
	bool CreateConnections();
	bool LoadNetwork(const wchar_t *wzFileName);
	void SaveNetwork(const wchar_t *wzFileName);
	void OutputNetworkInfo();
	void Clear();

	double CalculateCost(imagesample *pSample, uint8_t label);
	double BatchForward(imagesample* pSample, uint8_t label);
	void BatchBackward(double learnRate);
	//double Learn(imagesample *pSample, uint8_t label, uint32_t epoch, double rate);
	bool AccuracyTest(imagesample *pSample, uint8_t label, uint8_t *pbGuess);
	void PropagateForward(imagesample *pSample);
	void Serialize(ofstream &stream);

	bool _fInitialized = false;
	vector<Layer *> _vecLayers;
};