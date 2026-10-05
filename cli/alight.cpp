// alight: command-line front end of burnback-3d (https://codeberg.org/iff/burnback-3d, AGPL-3.0).
//
// Runs the solver's mesh reader and iteration kernels without the Qt GUI. The
// iteration sequence is the one in Actions::worker() (src/interface.cpp).
//
// Usage: alight mesh.json out_prefix [--cfl 1] [--iters 300] [--weight 1]
//               [--tol 0] [--json result.json]
// Writes out_prefix.u.f64 (burn time per node, raw little-endian doubles),
// out_prefix.meta.json and out_prefix.history.csv.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <src/headers/globals.h>
#include <src/headers/iosystem.h>
#include <src/headers/operations.h>

using namespace std;

#ifndef ALIGHT_VERSION
#define ALIGHT_VERSION "dev"
#endif

int main(int argc, char *argv[]) {
	if (argc == 2 && string(argv[1]) == "--version") {
		cout << "alight " << ALIGHT_VERSION << endl;
		return 0;
	}
	if (argc < 3 || string(argv[1]) == "--help") {
		cerr << "usage: alight mesh.json out_prefix [--cfl 1] [--iters 300] [--weight 1] [--tol 0]\n"
		        "              [--json result.json] [--expect-max value]\n"
		        "       alight --version\n"
		        "  --cfl      pseudo-time step factor (stable up to about 2 with --weight 1)\n"
		        "  --iters    maximum number of iterations\n"
		        "  --weight   diffusive weight; values below about 0.75 diverge\n"
		        "  --tol      stop when the largest nodal rate of change falls below this (0: run all iterations)\n"
		        "  --json     also write the GUI's result file (mesh plus burnbackResults)\n"
		        "  --expect-max  fail unless the largest burn time is within 5 % of this value\n";
		return 2;
	}
	string meshPath = argv[1];
	string prefix = argv[2];
	string jsonOut;
	double tol = 0, expectMax = -1;
	input.uInitial = 0;
	input.resume = false;
	input.cfl = 1;
	input.targetIter = 300;
	input.diffusiveWeight = 1;
	for (int i = 3; i + 1 < argc; i += 2) {
		string key = argv[i];
		if (key == "--cfl")
			input.cfl = atof(argv[i + 1]);
		else if (key == "--iters")
			input.targetIter = atoi(argv[i + 1]);
		else if (key == "--weight")
			input.diffusiveWeight = atof(argv[i + 1]);
		else if (key == "--tol")
			tol = atof(argv[i + 1]);
		else if (key == "--json")
			jsonOut = argv[i + 1];
		else if (key == "--expect-max")
			expectMax = atof(argv[i + 1]);
		else {
			cerr << "unknown option " << key << "\n";
			return 2;
		}
	}

	try {
		Json::readMesh(meshPath);
	} catch (std::exception &e) {
		cerr << "Error: " << e.what() << "\n";
		return 1;
	}
	const size_t nNodes = mesh.nodes.size();
	cout << "nodes " << nNodes << " tetrahedra " << mesh.tetrahedra.size() << " triangles " << mesh.triangles.size() << endl;

	currentIter = 0;
	timeTotal = 0;
	timeStep = 0;
	tetrahedraGeometry = TetrahedraGeometry(mesh.tetrahedra.size());
	angleTotal = vector<double>(nNodes);
	computationData = ComputationData(nNodes, mesh.tetrahedra.size());
	Geometry::computeGeometry();
	Nodes::setBoundaryConditions();
	if (anisotropic)
		Anisotropic::computeMatrix();
	maxRecession = Nodes::getMaxRecession();

	// minHeight is a sixth of the smallest tetrahedron altitude (Geometry::computeGeometry)
	timeStep = minHeight * input.cfl / maxRecession;
	cout << "time step " << timeStep << endl;

	size_t nInlet = 0;
	for (auto type : boundaryConditions)
		nInlet += type == INLET;
	if (nInlet == 0) {
		cerr << "Error: no inlet nodes\n";
		return 1;
	}

	ofstream history(prefix + ".history.csv");
	history << "iteration,error,max_rate,u_max\n";
	vector<double> previous(nNodes);
	double maxRate = INFINITY, error = 0;
	bool diverged = false;
	const auto start = chrono::steady_clock::now();
	for (; currentIter < input.targetIter; ++currentIter) {
		previous = computationData.uVertex;
		Tetrahedra::computeMeanGradient();
		Tetrahedra::computeVertexGradient();
		Nodes::applySymmetry();
		Tetrahedra::computeDiffusiveFlux();
		if (anisotropic)
			Anisotropic::computeRecession();
		Nodes::computeHamitonianFlux();
		Nodes::computeResults();
		error = Nodes::getError();

		// pseudo-time rate of change of the burn time; zero at the steady state
		maxRate = 0;
		double uMax = 0;
		for (size_t node = 0; node < nNodes; ++node) {
			maxRate = max(maxRate, abs(computationData.uVertex[node] - previous[node]) / timeStep);
			uMax = max(uMax, computationData.uVertex[node]);
		}
		history << currentIter + 1 << "," << error << "," << maxRate << "," << uMax << "\n";
		if ((currentIter + 1) % 100 == 0)
			cout << "iter " << currentIter + 1 << " error " << error << " max rate " << maxRate << " u max " << uMax << endl;
		if (error > 1 || !isfinite(maxRate)) {
			diverged = true;
			cerr << "Error: divergence detected at iteration " << currentIter + 1 << ". Reduce the CFL.\n";
			break;
		}
		if (tol > 0 && maxRate < tol) {
			++currentIter;
			break;
		}
	}
	const double seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();
	cout << "finished after " << currentIter << " iterations, max rate " << maxRate << ", " << seconds << " s" << endl;

	ofstream binary(prefix + ".u.f64", ios::binary);
	binary.write(reinterpret_cast<const char *>(computationData.uVertex.data()), nNodes * sizeof(double));

	ofstream meta(prefix + ".meta.json");
	meta.precision(12);
	meta << "{\"nodes\": " << nNodes << ", \"tetrahedra\": " << mesh.tetrahedra.size()
	     << ", \"iterations\": " << currentIter << ", \"time_step\": " << timeStep
	     << ", \"cfl\": " << input.cfl
	     << ", \"diffusive_weight\": " << input.diffusiveWeight << ", \"max_rate\": " << maxRate
	     << ", \"error\": " << error << ", \"diverged\": " << (diverged ? "true" : "false")
	     << ", \"converged\": " << ((tol > 0 && maxRate < tol) ? "true" : "false")
	     << ", \"seconds\": " << seconds << "}\n";

	if (!jsonOut.empty()) {
		bool pretty = false;
		Json::writeData(jsonOut, meshPath, pretty);
	}
	if (diverged)
		return 3;
	if (expectMax > 0) {
		// self-check used by the test suite: largest burn time within 5 % of the expected value
		const double uMax = *max_element(computationData.uVertex.begin(), computationData.uVertex.end());
		if (abs(uMax - expectMax) > 0.05 * expectMax) {
			cerr << "Error: largest burn time " << uMax << " differs from the expected " << expectMax << "\n";
			return 4;
		}
	}
	return 0;
}
