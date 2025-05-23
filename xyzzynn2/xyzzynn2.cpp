
#include "pch.h"
#include <iostream>
#include <filesystem>

int main(int argc, char **argv)
{
	Simulator sim;

	sim.Initialize();
	
	if (argc > 1)
	{
		if (_stricmp(argv[1], "-dump") == 0)
		{
			sim.DumpData();
			return 1;
		}
		else if (_stricmp(argv[1], "-create") == 0)
		{
			wchar_t filename[MAX_PATH];

			if (argc < 3)
			{
				wprintf(L"usage: -create <filename>\n");
				return 0;
			}

			swprintf_s(filename, _countof(filename), L"%S", argv[2]);
			sim.CreateNetwork();
			sim.SaveNetwork(filename);
			return 1;
		}
		else if (_stricmp(argv[1], "-learn") == 0)
		{
			wchar_t filename[MAX_PATH];


			if (argc < 5)
			{
				wprintf(L"usage: -learn <filename> <rate> <epochs>\n");
				return 0;
			}

			double rate = atof(argv[3]);
			int32_t epochs = atoi(argv[4]);
			int32_t trainLimit = 0;

			if (argc == 6)
			{
				trainLimit = atoi(argv[5]);
			}

			swprintf_s(filename, _countof(filename), L"%S", argv[2]);

			if (false == std::filesystem::exists(filename))
			{
				sim.CreateNetwork();
				sim.SaveNetwork(filename);
			}

			if (sim.LoadNetwork(filename))
			{
				sim.Learn(rate, epochs, trainLimit);
			}

			return 1;
		}
		else if (_stricmp(argv[1], "-test") == 0)
		{
			wchar_t filename[MAX_PATH];
			bool fdump = false;

			if (argc < 4)
			{
				wprintf(L"usage: -test <filename> [1|0]\n");
				return 0;
			}

			int d = atoi(argv[3]);

			swprintf_s(filename, _countof(filename), L"%S", argv[2]);
			sim.LoadNetwork(filename);
			sim.AccuracyTest(false, d != 0);
			return 1;
		
		}
	}

	wprintf(L" -create, -learn, -dump  or -test\n");
}
