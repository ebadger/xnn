#pragma once
#include "pch.h"

class Network;

class Simulator
{
public:
	void Initialize();
	void DumpData();	
	void CreateNetwork();
	double CalculateTotalCost();
	void Learn(double rate, int32_t epochs);
	void DumpNetwork();
	void AccuracyTest(bool fSaveMax, bool fDumpWrong);
	void DumpSample(imagesample *);

	bool LoadNetwork(const wchar_t *);
	void SaveNetwork(const wchar_t *);

private:
	FileMappingReadOnly<imagesample> _imagesTraining;
	FileMappingReadOnly<uint8_t> _labelsTraining;

	FileMappingReadOnly<imagesample> _imagesTest;
	FileMappingReadOnly<uint8_t> _labelsTest;

	Network *_pNetwork = nullptr;
};