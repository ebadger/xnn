#include "pch.h"
#include <random>
#include <cstring>

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

static inline float Sigmoidf(float x)
{
	return 1.0f / (1.0f + std::exp(-x));
}

void Network::ApplyActivation(uint32_t activation, float* pNeurons, uint32_t count)
{
	switch (activation)
	{
	case XnnAct_Sigmoid:
		for (uint32_t j = 0; j < count; ++j)
		{
			pNeurons[j] = Sigmoidf(pNeurons[j]);
		}
		break;
	case XnnAct_Relu:
		for (uint32_t j = 0; j < count; ++j)
		{
			float v = pNeurons[j];
			pNeurons[j] = v < 0.0f ? 0.0f : v;
		}
		break;
	case XnnAct_None:
	default:
		break;
	}
}

void Network::LoadInputLayer(const imagesample* pSample)
{
	const LayerDesc& in = _layers[0];
	float* pVal = _values.data() + in.neuronOffset;
	const uint8_t* pPx = pSample->pixels;
	const uint32_t n = in.neuronCount;
	CheckConditionFailFast(n == 784);
	// Same normalization as the original code (Utils::Relu(pixel, 255)) which
	// is just pixel/255 since pixels are unsigned bytes.
	const float inv = 1.0f / 255.0f;
	for (uint32_t i = 0; i < n; ++i)
	{
		pVal[i] = (float)pPx[i] * inv;
	}
}

// ----------------------------------------------------------------------------
// Construction / topology
// ----------------------------------------------------------------------------

bool Network::AddLayer(uint32_t neurons, double /*multiplier*/)
{
	if (_fInitialized)
	{
		return false;
	}

	LayerDesc d{};
	d.neuronCount     = neurons;
	d.neuronOffset    = (uint32_t)_values.size();        // before resize
	d.prevNeuronCount = _layers.empty() ? 0u : _layers.back().neuronCount;
	d.weightOffset    = (uint32_t)_weights.size();       // before resize
	d.activation      = _layers.empty() ? XnnAct_None : XnnAct_Sigmoid;

	const size_t weightsForThisLayer = (size_t)neurons * (size_t)d.prevNeuronCount;
	_weights.resize(_weights.size() + weightsForThisLayer, 0.0f);
	_biases.resize(_biases.size() + neurons, 0.0f);
	_values.resize(_values.size() + neurons, 0.0f);
	_deltas.resize(_deltas.size() + neurons, 0.0f);

	_layers.push_back(d);
	return true;
}

bool Network::CreateConnections()
{
	if (_fInitialized)
	{
		return false;
	}

	// Xavier/Glorot uniform: limit = sqrt(6 / (fan_in + fan_out)).
	std::mt19937 rng(std::random_device{}());

	for (size_t l = 1; l < _layers.size(); ++l)
	{
		const LayerDesc& curr = _layers[l];
		const uint32_t fanIn  = curr.prevNeuronCount;
		const uint32_t fanOut = curr.neuronCount;
		const float limit = (float)std::sqrt(6.0 / (double)(fanIn + fanOut));
		std::uniform_real_distribution<float> dist(-limit, limit);

		float* W = _weights.data() + curr.weightOffset;
		const size_t count = (size_t)fanOut * (size_t)fanIn;
		for (size_t k = 0; k < count; ++k)
		{
			W[k] = dist(rng);
		}
		// Biases default to zero (already zero-initialized above).
	}

	_fInitialized = true;
	return true;
}

void Network::Clear()
{
	_layers.clear();
	_weights.clear();
	_biases.clear();
	_values.clear();
	_deltas.clear();
	_fInitialized = false;
}

void Network::OutputNetworkInfo()
{
	wprintf(L"Layers: %u [", (uint32_t)_layers.size());
	for (const LayerDesc& d : _layers)
	{
		wprintf(L" %u ", d.neuronCount);
	}
	wprintf(L"]\r\n");
	wprintf(L"  totalNeurons=%zu  totalWeights=%zu  bytes(weights)=%zu\r\n",
	        _values.size(), _weights.size(), _weights.size() * sizeof(float));
}

// ----------------------------------------------------------------------------
// File I/O
// ----------------------------------------------------------------------------

bool Network::LoadNetwork(const wchar_t* wzFileName)
{
	Clear();

	std::ifstream is;
	is.open(wzFileName, std::ios::in | std::ios::binary);
	if (!is.is_open())
	{
		WCHAR cdir[MAX_PATH];
		GetCurrentDirectoryW(MAX_PATH, cdir);
		wprintf(L"Failed to open %s in dir: %s\r\n", wzFileName, cdir);
		return false;
	}

	XnnHeader hdr{};
	is.read((char*)&hdr, sizeof(hdr));
	if (!is || hdr.magic != XNN_MAGIC)
	{
		wprintf(L"%s: not an XNN1 file (magic=0x%08x)\r\n", wzFileName, hdr.magic);
		return false;
	}
	if (hdr.version != XNN_VERSION || hdr.dtype != XNN_DTYPE_F32)
	{
		wprintf(L"%s: unsupported version=%u dtype=%u\r\n",
		        wzFileName, hdr.version, hdr.dtype);
		return false;
	}

	_layers.resize(hdr.layerCount);
	is.read((char*)_layers.data(), (std::streamsize)(sizeof(LayerDesc) * hdr.layerCount));
	if (!is)
	{
		wprintf(L"%s: truncated layer table\r\n", wzFileName);
		Clear();
		return false;
	}

	_weights.resize((size_t)hdr.totalWeights);
	_biases.resize((size_t)hdr.totalNeurons);
	_values.resize((size_t)hdr.totalNeurons, 0.0f);
	_deltas.resize((size_t)hdr.totalNeurons, 0.0f);

	is.seekg((std::streamoff)hdr.weightsOffset, std::ios::beg);
	is.read((char*)_weights.data(), (std::streamsize)(_weights.size() * sizeof(float)));
	if (!is)
	{
		wprintf(L"%s: truncated weight blob\r\n", wzFileName);
		Clear();
		return false;
	}

	is.seekg((std::streamoff)hdr.biasesOffset, std::ios::beg);
	is.read((char*)_biases.data(), (std::streamsize)(_biases.size() * sizeof(float)));
	if (!is)
	{
		wprintf(L"%s: truncated bias blob\r\n", wzFileName);
		Clear();
		return false;
	}

	_fInitialized = true;
	wprintf(L"loaded: %s\r\n", wzFileName);
	OutputNetworkInfo();
	return true;
}

void Network::SaveNetwork(const wchar_t* wzFileName)
{
	std::ofstream os;
	os.open(wzFileName, std::ios::out | std::ios::binary | std::ios::trunc);
	if (!os.is_open())
	{
		wprintf(L"failed to open for write: %s\r\n", wzFileName);
		return;
	}

	XnnHeader hdr{};
	hdr.magic        = XNN_MAGIC;
	hdr.version      = XNN_VERSION;
	hdr.dtype        = XNN_DTYPE_F32;
	hdr.layerCount   = (uint32_t)_layers.size();
	hdr.totalNeurons = (uint64_t)_biases.size();
	hdr.totalWeights = (uint64_t)_weights.size();

	const uint64_t afterTable =
		(uint64_t)sizeof(XnnHeader) + (uint64_t)sizeof(LayerDesc) * hdr.layerCount;
	const uint64_t aligned = (afterTable + (XNN_PAGE - 1)) & ~(uint64_t)(XNN_PAGE - 1);
	hdr.weightsOffset = aligned;
	hdr.biasesOffset  = aligned + hdr.totalWeights * sizeof(float);

	os.write((const char*)&hdr, sizeof(hdr));
	os.write((const char*)_layers.data(), (std::streamsize)(sizeof(LayerDesc) * _layers.size()));

	// Pad to page-aligned weights offset.
	const uint64_t padBytes = hdr.weightsOffset - afterTable;
	if (padBytes > 0)
	{
		std::vector<char> zero((size_t)padBytes, 0);
		os.write(zero.data(), (std::streamsize)padBytes);
	}

	os.write((const char*)_weights.data(), (std::streamsize)(_weights.size() * sizeof(float)));
	os.write((const char*)_biases.data(),  (std::streamsize)(_biases.size()  * sizeof(float)));

	wprintf(L"saved: %s\r\n", wzFileName);
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------

void Network::PropagateForward(imagesample* /*pSample*/)
{
	// Caller is expected to have already loaded inputs via LoadInputLayer().
	// Walk forward through the dense weight matrices.
	for (size_t l = 1; l < _layers.size(); ++l)
	{
		const LayerDesc& curr = _layers[l];
		const LayerDesc& prev = _layers[l - 1];

		const float* W      = _weights.data() + curr.weightOffset;
		const float* prevA  = _values.data()  + prev.neuronOffset;
		const float* bias   = _biases.data()  + curr.neuronOffset;
		float*       out    = _values.data()  + curr.neuronOffset;

		const uint32_t fanOut = curr.neuronCount;
		const uint32_t fanIn  = curr.prevNeuronCount;

		// out[j] = bias[j] + sum_i W[j*fanIn + i] * prevA[i]
		for (uint32_t j = 0; j < fanOut; ++j)
		{
			const float* Wrow = W + (size_t)j * fanIn;
			float sum = bias[j];
			for (uint32_t i = 0; i < fanIn; ++i)
			{
				sum += Wrow[i] * prevA[i];
			}
			out[j] = sum;
		}

		ApplyActivation(curr.activation, out, fanOut);
	}
}

double Network::CalculateCost(imagesample* pSample, uint8_t label)
{
	LoadInputLayer(pSample);
	PropagateForward(pSample);

	const LayerDesc& outL = _layers.back();
	const float* out = _values.data() + outL.neuronOffset;
	double cost = 0.0;
	for (uint32_t j = 0; j < outL.neuronCount; ++j)
	{
		double expected = (label == (uint8_t)j) ? 1.0 : 0.0;
		double d = expected - (double)out[j];
		cost += d * d;
	}
	return cost;
}

double Network::BatchForward(imagesample* pSample, uint8_t label)
{
	LoadInputLayer(pSample);
	PropagateForward(pSample);

	const LayerDesc& outL = _layers.back();
	const float* out = _values.data() + outL.neuronOffset;

	double total = 0.0;
	for (uint32_t j = 0; j < outL.neuronCount; ++j)
	{
		double expected = (label == (uint8_t)j) ? 1.0 : 0.0;
		double d = expected - (double)out[j];
		total += d * d;
	}
	return total;
}

bool Network::AccuracyTest(imagesample* pSample, uint8_t label, uint8_t* pbGuess)
{
	*pbGuess = 0xFF;

	LoadInputLayer(pSample);
	PropagateForward(pSample);

	const LayerDesc& outL = _layers.back();
	const float* out = _values.data() + outL.neuronOffset;

	uint32_t maxJ = 0;
	float    maxV = -std::numeric_limits<float>::infinity();
	for (uint32_t j = 0; j < outL.neuronCount; ++j)
	{
		if (out[j] > maxV)
		{
			maxV = out[j];
			maxJ = j;
		}
	}

	*pbGuess = (uint8_t)maxJ;
	return label == (uint8_t)maxJ;
}

// ----------------------------------------------------------------------------
// Backprop
// ----------------------------------------------------------------------------

void Network::PropagateBackward(uint8_t label, double rate)
{
	const float fRate = (float)rate;

	// 1) Output-layer delta: dL/da = (a - y); for sigmoid, sigma'(z) = a*(1-a).
	{
		const LayerDesc& outL = _layers.back();
		float* a = _values.data() + outL.neuronOffset;
		float* d = _deltas.data() + outL.neuronOffset;
		for (uint32_t j = 0; j < outL.neuronCount; ++j)
		{
			float y = (label == (uint8_t)j) ? 1.0f : 0.0f;
			float aj = a[j];
			d[j] = (aj - y) * aj * (1.0f - aj);
		}
	}

	// 2) Hidden-layer deltas, propagating from output back toward input.
	//    We only go down to layer index 1 (input layer has no delta).
	for (int l = (int)_layers.size() - 2; l >= 1; --l)
	{
		const LayerDesc& curr = _layers[(size_t)l];
		const LayerDesc& next = _layers[(size_t)l + 1];

		const float* Wnext      = _weights.data() + next.weightOffset; // [next.fanOut x next.fanIn=curr.neuronCount]
		const float* nextDelta  = _deltas.data()  + next.neuronOffset;
		const float* currVal    = _values.data()  + curr.neuronOffset;
		float*       currDelta  = _deltas.data()  + curr.neuronOffset;

		const uint32_t nFanOut = next.neuronCount;
		const uint32_t nFanIn  = next.prevNeuronCount;   // == curr.neuronCount

		// Zero this layer's deltas, then scatter-add. Iterating the outer loop
		// over j (next-layer neurons) keeps Wnext access sequential.
		std::memset(currDelta, 0, sizeof(float) * curr.neuronCount);
		for (uint32_t j = 0; j < nFanOut; ++j)
		{
			const float dj = nextDelta[j];
			const float* Wrow = Wnext + (size_t)j * nFanIn;
			for (uint32_t i = 0; i < nFanIn; ++i)
			{
				currDelta[i] += Wrow[i] * dj;
			}
		}

		// Apply sigmoid derivative in place.
		for (uint32_t i = 0; i < curr.neuronCount; ++i)
		{
			float v = currVal[i];
			currDelta[i] *= v * (1.0f - v);
		}
	}

	// 3) Apply weight + bias updates everywhere except the input layer.
	for (size_t l = 1; l < _layers.size(); ++l)
	{
		const LayerDesc& curr = _layers[l];
		const LayerDesc& prev = _layers[l - 1];

		float*       W      = _weights.data() + curr.weightOffset;
		float*       bias   = _biases.data()  + curr.neuronOffset;
		const float* delta  = _deltas.data()  + curr.neuronOffset;
		const float* prevA  = _values.data()  + prev.neuronOffset;

		const uint32_t fanOut = curr.neuronCount;
		const uint32_t fanIn  = curr.prevNeuronCount;

		for (uint32_t j = 0; j < fanOut; ++j)
		{
			const float scale = fRate * delta[j];
			float* Wrow = W + (size_t)j * fanIn;
			// Streaming write pattern: each weight written once, contiguous.
			for (uint32_t i = 0; i < fanIn; ++i)
			{
				Wrow[i] -= scale * prevA[i];
			}
			bias[j] -= scale;
		}
	}
}
