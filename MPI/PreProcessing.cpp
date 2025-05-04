#include <metis.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>

using namespace std;

int main()
{
    string filename = "../Dataset/Mini_Data.txt"; // Path to the dataset file

    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not open the file." << endl;
        return -1;
    }

    // Read and display the first line (number of nodes and edges)
    string firstLine;
    if (getline(file, firstLine))
    {
        cout << "File header (number of nodes and edges): " << firstLine << endl;
    }
    else
    {
        cerr << "Error: File is empty or could not read the first line." << endl;
        return -1;
    }

    vector<idx_t> xadj;     // Adjacency index (CSR format)
    vector<idx_t> adjncy;   // Adjacency list (CSR format)
    vector<idx_t> adjwgt;   // Edge weights (optional for METIS)
    vector<idx_t> vertices; // List of all vertices for partitioning

    idx_t nvtxs = 0;  // Number of vertices
    idx_t nedges = 0; // Number of edges
    string line;

    // Read the dataset and construct the graph in CSR format for METIS
    while (getline(file, line))
    {
        istringstream iss(line);
        idx_t from, to, weight;
        if (!(iss >> from >> to >> weight))
        {
            cerr << "Error: Invalid line format." << endl;
            return -1;
        }

        // Update the number of vertices
        nvtxs = max(nvtxs, max(from, to));

        // Add edges to adjacency list (convert to 0-based indexing)
        adjncy.push_back(to - 1);
        adjwgt.push_back(weight);

        adjncy.push_back(from - 1); // Add reverse edge for undirected graph
        adjwgt.push_back(weight);

        nedges++;
    }

    file.close();

    // Construct the xadj array (CSR format)
    xadj.resize(nvtxs + 1, 0);
    for (size_t i = 0; i < adjncy.size(); ++i)
    {
        xadj[adjncy[i] + 1]++;
    }
    for (size_t i = 1; i <= nvtxs; ++i)
    {
        xadj[i] += xadj[i - 1];
    }

    // Print the constructed CSR graph (for debugging purposes)
    cout << "Constructed CSR graph:\n";
    cout << "xadj: ";
    for (const auto& val : xadj)
    {
        cout << val << " ";
    }
    cout << "\nadjncy: ";
    for (const auto& val : adjncy)
    {
        cout << val << " ";
    }
    cout << "\nadjwgt: ";
    for (const auto& val : adjwgt)
    {
        cout << val << " ";
    }
    cout << endl;

    // Define METIS parameters
    idx_t ncon = 1; // Number of balancing constraints
    idx_t nparts = 2; // Number of partitions (can be adjusted as needed)
    idx_t objval; // Edge-cut or communication volume
    vector<idx_t> part(nvtxs); // Partition vector

    // Call METIS_PartGraphKway
    int status = METIS_PartGraphKway(
        &nvtxs, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, adjwgt.data(),
        &nparts, nullptr, nullptr, nullptr, &objval, part.data());

    if (status == METIS_OK)
    {
        cout << "METIS_PartGraphKway completed successfully.\n";
        cout << "Edge-cut: " << objval << "\n";
        cout << "Partitioning result:\n";
        for (size_t i = 0; i < part.size(); ++i)
        {
            cout << "Vertex " << i + 1 << " -> Partition " << part[i] << "\n";
        }
    }
    else
    {
        cerr << "Error: METIS_PartGraphKway failed with status " << status << ".\n";
        return -1;
    }

    return 0;
}
