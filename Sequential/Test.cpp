//////  Compile by :  g++ -o Test Test.cpp -lmetis

#include <metis.h>
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // Create a vector of integers
    int n = 6; // number of vertices
    int ncon = 1; // number of constraints
    int nparts = 2; // number of partitions
    vector<int> xadj = {0, 2, 4, 6, 8, 10, 12}; // adjacency list
    vector<int> adjncy = {1, 2, 0, 3, 4, 1, 5, 2, 3, 4, 5}; // adjacency list
    idx_t * part = new idx_t[n]; // allocate memory for partitioning result
    idx_t* val;
   
    METIS_PartGraphKway(
        &n, // number of vertices
        &ncon, // number of constraints
        xadj.data(), // vertex adjacency list
        adjncy.data(), // adjacency list
        nullptr, // edge weights (not used)
        nullptr, // vertex weights (not used)
        nullptr, // options for METIS
        &nparts, // number of partitions
        NULL, // options for METIS
        NULL, // objective value (output)
        nullptr, // edge weights (not used)
        val, // vertex weights (not used)
        part // partitioning result (output)
    );

    cout << "Partitioning result:" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "Vertex " << i << " is in partition " << part[i] << endl;
    }   
}