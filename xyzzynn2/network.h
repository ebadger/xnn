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
	void PropagateBackward(uint8_t label, double rate);

	bool AccuracyTest(imagesample *pSample, uint8_t label, uint8_t *pbGuess);
	void PropagateForward(imagesample *pSample);
	void Serialize(ofstream &stream);

	bool _fInitialized = false;
	vector<Layer *> _vecLayers;
};