#pragma once
#include "pch.h"

// Activation kind for each layer.
enum XnnActivation : uint32_t
{
	XnnAct_None    = 0,   // input layer
	XnnAct_Sigmoid = 1,
	XnnAct_Relu    = 2,
};

// Per-layer descriptor. Plain-old-data, fixed size, file-format compatible.
// All offsets are *element* offsets into the parallel arrays owned by Network.
struct LayerDesc
{
	uint32_t neuronCount;        // number of activations in this layer
	uint32_t neuronOffset;       // start index into _biases / _values / _deltas
	uint32_t prevNeuronCount;    // 0 for input layer
	uint32_t weightOffset;       // start index into _weights (0 for input layer)
	uint32_t activation;         // XnnActivation
	uint32_t reserved[3];        // pad to 32 bytes for future use
};
static_assert(sizeof(LayerDesc) == 32, "LayerDesc must be 32 bytes");

// On-disk header for the .xnn file format ("XNN1").
//
// Layout:
//   [ XnnHeader                           ]
//   [ LayerDesc[layerCount]               ]
//   [ pad to weightsOffset (page aligned) ]
//   [ float32 weights[totalWeights]       ]
//   [ float32 biases [totalNeurons]       ]
//
// All integers are little-endian. Native-endian floats (we only target x64 / ARM64
// little-endian Windows). The weight blob is page-aligned (4096) so the file can
// be memory-mapped and the matmul kernels can use aligned SIMD loads.
struct XnnHeader
{
	uint32_t magic;              // 'X','N','N','1' = 0x314E4E58 little-endian
	uint32_t version;            // 1
	uint32_t dtype;              // 0 = float32 (only value supported today)
	uint32_t layerCount;
	uint64_t totalNeurons;       // sum of all layer neuron counts
	uint64_t totalWeights;       // sum of all (neuronCount * prevNeuronCount)
	uint64_t weightsOffset;      // file offset, page-aligned
	uint64_t biasesOffset;       // file offset
	uint64_t reserved[2];        // pad to 64 bytes
};
static_assert(sizeof(XnnHeader) == 64, "XnnHeader must be 64 bytes");

constexpr uint32_t XNN_MAGIC   = 0x314E4E58u; // 'X','N','N','1' little-endian
constexpr uint32_t XNN_VERSION = 1u;
constexpr uint32_t XNN_DTYPE_F32 = 0u;
constexpr size_t   XNN_PAGE = 4096;

// Network: Struct-of-Arrays multilayer perceptron.
//
// All neuron state and all weights live in a handful of contiguous float32
// buffers. There are no Neuron / Connection objects on the heap. The forward
// pass is a sequence of dense GEMV operations; backprop is GEMV + AXPY-style
// weight updates -- both auto-vectorize and prefetch cleanly.
class Network
{
public:
	bool AddLayer(uint32_t neurons, double /*multiplier*/ = 0.0);
	bool CreateConnections();

	bool LoadNetwork(const wchar_t* wzFileName);
	void SaveNetwork(const wchar_t* wzFileName);
	void OutputNetworkInfo();
	void Clear();

	double CalculateCost(imagesample* pSample, uint8_t label);
	double BatchForward(imagesample* pSample, uint8_t label);
	void   PropagateForward(imagesample* pSample);
	void   PropagateBackward(uint8_t label, double rate);
	bool   AccuracyTest(imagesample* pSample, uint8_t label, uint8_t* pbGuess);

	// SoA storage. Public for the simulator's debug dump.
	std::vector<LayerDesc> _layers;
	std::vector<float>     _weights;   // dense per-layer matrices, row-major [j, i]
	std::vector<float>     _biases;    // one per neuron across all layers
	std::vector<float>     _values;    // current activations
	std::vector<float>     _deltas;    // backprop error term per neuron

	bool _fInitialized = false;

private:
	void LoadInputLayer(const imagesample* pSample);
	void ApplyActivation(uint32_t activation, float* pNeurons, uint32_t count);
};
