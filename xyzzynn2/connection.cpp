#include "pch.h"
#include <random>

Connection::Connection(double weight)
{
	_weight = weight;
}

Connection::Connection()
{
	_weight = Utils::RandomDouble(0.0000001,1.0);
}

Connection::~Connection()
{
}

void Connection::Serialize(ofstream &stream)
{
	stream.write((const char *)&_weight, sizeof(double));
}

void Connection::DeSerialize(ifstream &stream)
{
	stream.read((char *)&_weight, sizeof(double));
}