#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono> // Add chrono for timing

using namespace std;
using namespace std::chrono; // Add namespace for chrono

// Edge structure for updates
struct Edge {
    int u, v;
    double weight;
};

// Graph structure in CSR format
struct Graph {
    vector<int> adj;        // CSR adjacency list (edge endpoints)
    vector<int> row_ptr;    // CSR row pointers
    vector<double> weights; // Edge weights
    vector<double> dist;    // Distance from source
    vector<int> parent;     // Parent in SSSP tree
    vector<vector<int>> children; // Children in SSSP tree (used for batch deletion)
    int n;                  // Number of vertices
    int m;                  // Number of edges

    Graph(int n, int m, const vector<int>& src, const vector<int>& dst, const vector<double>& weights) {
        try {
            if (n <= 0 || m <= 0) {
                throw runtime_error("Invalid graph size");
            }

            this->n = n;
            this->m = m * 2; // Double for undirected edges

            // Resize vectors for 1-based indexing
            dist.resize(n + 1, INFINITY);
            parent.resize(n + 1, -1);
            children.resize(n + 1); // Resized, but only built when needed
            row_ptr.resize(n + 2, 0);
            adj.resize(this->m);
            this->weights.resize(this->m);

            // Build CSR with 1-based indexing
            vector<int> edge_count(n + 1, 0);
            for (int i = 0; i < m; i++) {
                // Add bounds check during count calculation
                if (src[i] < 1 || src[i] > n || dst[i] < 1 || dst[i] > n) {
                     throw runtime_error("Edge vertex index out of bounds during CSR build");
                }
                edge_count[src[i]]++;
                edge_count[dst[i]]++;
            }

            row_ptr[1] = 0;
            for (int i = 1; i <= n; i++) {
                row_ptr[i + 1] = row_ptr[i] + edge_count[i];
                // Check potential overflow during row_ptr calculation
                if (row_ptr[i + 1] > this->m) {
                     throw runtime_error("CSR index calculation potentially out of bounds");
                }
            }
             // Final check after loop
            if (row_ptr[n + 1] != this->m) {
                 throw runtime_error("CSR final edge count mismatch");
            }


            vector<int> current_pos(n + 1, 0);
            for (int i = 0; i < m; i++) {
                // Forward edge
                int u_node = src[i];
                int v_node = dst[i];
                double w = weights[i];

                // Bounds check before accessing CSR arrays
                if (u_node < 1 || u_node > n || v_node < 1 || v_node > n) {
                     throw runtime_error("Edge vertex index out of bounds during CSR population");
                }


                int pos_fwd = row_ptr[u_node] + current_pos[u_node]++;
                 if (pos_fwd < 0 || pos_fwd >= this->m) {
                     throw runtime_error("CSR forward edge index out of bounds");
                 }
                adj[pos_fwd] = v_node;
                this->weights[pos_fwd] = w;

                // Reverse edge
                int pos_rev = row_ptr[v_node] + current_pos[v_node]++;
                 if (pos_rev < 0 || pos_rev >= this->m) {
                     throw runtime_error("CSR reverse edge index out of bounds");
                 }
                adj[pos_rev] = u_node;
                this->weights[pos_rev] = w;
            }
        } catch (const exception& e) {
            cerr << "Error creating graph: " << e.what() << endl;
            throw;
        }
    }

    // Build children list based on parent array (used for batch deletion)
    void buildChildren() {
        children.assign(n + 1, vector<int>()); // Use n+1 for 1-based indexing
        for (int i = 1; i <= n; i++) { // Iterate from 1 to n
            if (parent[i] != -1 && parent[i] >= 1 && parent[i] <= n) { // Add bounds check for parent index
                children[parent[i]].push_back(i);
            }
        }
    }
};

// Custom comparator for min-heap priority queue
struct Compare {
    bool operator()(const pair<double, int>& a, const pair<double, int>& b) {
        return a.first > b.first; // Min-heap based on distance
    }
};

// Mark descendants as affected (DFS) - Used in Batch Update
void markDescendants(Graph& g, int v, priority_queue<pair<double, int>, vector<pair<double, int>>, Compare>& pq) {
    // Check bounds before accessing graph data
    if (v < 1 || v > g.n) return;

    g.dist[v] = INFINITY;
    g.parent[v] = -1;
    pq.push({g.dist[v], v}); // Push infinity to ensure reprocessing if needed

    // Check bounds before accessing children
    if (v >= 1 && v < g.children.size()) {
        for (int child : g.children[v]) {
             // Recursive call already checks bounds
            markDescendants(g, child, pq);
        }
    }
     // Clear children list for the current node after processing
     // to avoid issues if buildChildren is called again later
     if (v >= 1 && v < g.children.size()) {
         g.children[v].clear();
     }
}


// Update SSSP for multiple edge changes (Batch Update)
void updateSSSPBatch(Graph& g, const vector<Edge>& changes, const vector<bool>& isInsertion) {
    // Priority queue for vertices with their distances
    priority_queue<pair<double, int>, vector<pair<double, int>>, Compare> pq;

    // Build children list for deletion handling
    g.buildChildren(); // Needed for markDescendants

    // Step 1: Process deletions first
    for (size_t i = 0; i < changes.size(); i++) {
        if (!isInsertion[i]) {
            int u = changes[i].u, v = changes[i].v;
             // Bounds check
             if (u < 1 || u > g.n || v < 1 || v > g.n) continue;

            // Identify which vertex might be disconnected if the edge was in the tree
            int x = -1, y = -1;
             if (g.dist[u] != INFINITY && g.dist[v] != INFINITY) { // Only consider if both are reachable
                 x = (g.dist[u] > g.dist[v]) ? u : v;
                 y = (x == u) ? v : u;
             } else if (g.dist[u] != INFINITY) { // If only u is reachable, v might depend on u
                 x = v; y = u;
             } else if (g.dist[v] != INFINITY) { // If only v is reachable, u might depend on v
                 x = u; y = v;
             }


            // If x is valid and its parent was y, the edge was in the SSSP tree
            if (x != -1 && g.parent[x] == y) {
                markDescendants(g, x, pq);
            }
        }
    }

    // Step 2: Process insertions
    for (size_t i = 0; i < changes.size(); i++) {
        if (isInsertion[i]) {
            int u = changes[i].u, v = changes[i].v;
            double w = changes[i].weight;
             // Bounds check
             if (u < 1 || u > g.n || v < 1 || v > g.n || w < 0) continue; // Ignore invalid edges/weights


            // Check if the new edge offers a shorter path from u to v
            if (g.dist[u] != INFINITY && g.dist[u] + w < g.dist[v]) {
                g.dist[v] = g.dist[u] + w;
                g.parent[v] = u;
                pq.push({g.dist[v], v});
            }
            // Check if the new edge offers a shorter path from v to u
            if (g.dist[v] != INFINITY && g.dist[v] + w < g.dist[u]) {
                g.dist[u] = g.dist[v] + w;
                g.parent[u] = v;
                pq.push({g.dist[u], u});
            }
        }
    }

    // Step 3: Update affected subgraph using Dijkstra-like relaxation
    while (!pq.empty()) {
        auto top = pq.top();
        double d = top.first; // Use 'd' to avoid conflict with g.dist
        int z = top.second;
        pq.pop();

        // Bounds check
        if (z < 1 || z > g.n) continue;

        // Skip if a shorter path to z was already found (stale entry in PQ)
        // Use a small epsilon for floating-point comparison
        if (d > g.dist[z] + 1e-9) continue;

        // Relax edges outgoing from z
        for (int i = g.row_ptr[z]; i < g.row_ptr[z + 1]; i++) {
             // Bounds check for adjacency list access
             if (i < 0 || i >= g.adj.size()) continue; // Should not happen with correct CSR


            int neighbor = g.adj[i];
            double weight = g.weights[i];

             // Bounds check for neighbor
             if (neighbor < 1 || neighbor > g.n || weight < 0) continue;


            // If a shorter path to neighbor is found through z
            if (g.dist[z] != INFINITY && g.dist[z] + weight < g.dist[neighbor]) {
                g.dist[neighbor] = g.dist[z] + weight;
                g.parent[neighbor] = z;
                pq.push({g.dist[neighbor], neighbor});
            }
        }
    }
}

void dijkstra(Graph& g, int source) {
    try {
        if (source < 1 || source > g.n) {
            throw runtime_error("Invalid source vertex");
        }

        fill(g.dist.begin(), g.dist.end(), INFINITY);
        fill(g.parent.begin(), g.parent.end(), -1);

        priority_queue<pair<double, int>, vector<pair<double, int>>, Compare> pq;
        // No need for explicit visited array in standard Dijkstra with PQ
        // if we check for stale entries (d > g.dist[u])

        g.dist[source] = 0.0;
        pq.push({0.0, source});

        while (!pq.empty()) {
            double d = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            // If we found a shorter path already, skip processing this element
            if (d > g.dist[u] + 1e-9) { // Use epsilon for float comparison
                 continue;
            }


            // Relax neighbors
            for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                 // Bounds check for CSR access
                 if (i < 0 || i >= g.adj.size()) continue;


                int v = g.adj[i];
                double w = g.weights[i];

                 // Bounds check for neighbor and weight
                 if (v < 1 || v > g.n || w < 0) continue;


                // If a shorter path to v is found through u
                if (g.dist[u] != INFINITY && g.dist[u] + w < g.dist[v]) {
                    g.dist[v] = g.dist[u] + w;
                    g.parent[v] = u;
                    pq.push({g.dist[v], v});
                }
            }
        }
    } catch (const exception& e) {
        cerr << "Error in Dijkstra's algorithm: " << e.what() << endl;
        throw;
    }
}

// Function to load graph data from a file
void load_data(string filename, int& n, int& m, vector<int>& src, vector<int>& dst, vector<double>& weights) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Error opening file: " + filename);
    }

    if (!(file >> n >> m)) {
         file.close();
         throw runtime_error("Error reading graph dimensions (n, m) from file: " + filename);
     }


     if (n <= 0 || m < 0) { // Allow m=0 for empty graphs
         file.close();
         throw runtime_error("Invalid graph dimensions read from file (n=" + to_string(n) + ", m=" + to_string(m) + ")");
     }


    src.resize(m);
    dst.resize(m);
    weights.resize(m);

    int max_vertex_id = 0;
    for (int i = 0; i < m; i++) {
         if (!(file >> src[i] >> dst[i] >> weights[i])) {
             file.close();
             throw runtime_error("Error reading edge data at index " + to_string(i) + " from file: " + filename);
         }
        // Keep track of the highest vertex ID encountered
        max_vertex_id = max({max_vertex_id, src[i], dst[i]});
    }

    file.close(); // Close file promptly after reading

    // Check if the highest vertex ID exceeds the declared number of vertices 'n'
    if (max_vertex_id > n) {
        cerr << "Warning: Max vertex ID encountered (" << max_vertex_id
             << ") is greater than the number of vertices specified in the file (" << n
             << "). Adjusting n to " << max_vertex_id << "." << endl;
        n = max_vertex_id; // Adjust n to the actual maximum vertex ID found
    }
     // Add a check for 0 or negative vertex IDs if using 1-based indexing
     for (int i = 0; i < m; ++i) {
         if (src[i] <= 0 || dst[i] <= 0) {
             throw runtime_error("Invalid vertex ID (<= 0) found in edge data. Ensure 1-based indexing.");
         }
     }
}

void load_updates(const string& filename, vector<Edge>& changes, vector<bool>& isInsertion) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Error opening updates file: " + filename);
    }

    int num_updates;
    file >> num_updates;

    string type;
    int u, v;
    double w;
    while (file >> u >> v >> w >> type) {
        if (type == "1" || type == "i") {
            changes.push_back({u, v, w});
            isInsertion.push_back(true);
        } else if (type == "0" || type == "d") {
            changes.push_back({u, v, w});
            isInsertion.push_back(false);
        } else {
            cerr << "Unknown update type in updates file: " << type << endl;
        }
    }
    file.close();
}



int main() {
    try {
        // Start timing the whole program
        auto start_total = high_resolution_clock::now();

        int n, m;
        vector<int> src, dst;
        vector<double> weights;

        string filename = "../Dataset/sample_graph.txt"; 
        try {
            auto start_load = high_resolution_clock::now();
            load_data(filename, n, m, src, dst, weights);
            auto end_load = high_resolution_clock::now();
            cout << "Loaded graph with " << n << " vertices and " << m << " edges in "
                 << duration_cast<milliseconds>(end_load - start_load).count() << "ms" << endl;

            // Verify data immediately after loading (redundant with check in Graph constructor, but safe)
            for (int i = 0; i < m; i++) {
                if (src[i] <= 0 || src[i] > n || dst[i] <= 0 || dst[i] > n) {
                    cerr << "Invalid edge in input data: (" << src[i] << ", " << dst[i] << ")" << endl;
                    throw runtime_error("Edge endpoints out of range in input file");
                }
                 if (weights[i] < 0) {
                     cerr << "Warning: Negative edge weight found in input: " << weights[i] << ". Dijkstra assumes non-negative weights." << endl;
                                      }
            }

            // Initialize graph with robust error handling
            auto start_init = high_resolution_clock::now();
            Graph g(n, m, src, dst, weights);
            auto end_init = high_resolution_clock::now();
            cout << "Graph construction took "
                 << duration_cast<milliseconds>(end_init - start_init).count() << "ms" << endl;

          
            int print_count = min(10, n);
            
            int source_node = 1; 
            cout << "Running initial Dijkstra from source vertex " << source_node << "..." << endl;
            dijkstra(g, source_node);


            vector<Edge> changes;
            vector<bool> isInsertion;

            //Load updates from file
            string updates_filename = "../Dataset/updates.txt";

            load_updates(updates_filename, changes, isInsertion);

             // Verify update vertices are in range
             cout << "Applying batch changes:" << endl;
             for (size_t i = 0; i < changes.size(); ++i) {
                 const auto& edge = changes[i];
                 //cout << "  Change " << i << ": (" << edge.u << ", " << edge.v << ", " << edge.weight << ") - " << (isInsertion[i] ? "Insert" : "Delete") << endl;
                 if (edge.u <= 0 || edge.u > n || edge.v <= 0 || edge.v > n) {
                     cerr << "Invalid update edge: (" << edge.u << ", " << edge.v << ")" << endl;
                     throw runtime_error("Update edge endpoints out of range");
                 }
             }


            cout << "\nBefore batch updates:\n";
            for (int i = 1; i <= print_count; i++) {
                 cout << "Vertex " << i << ": Dist = " << (g.dist[i] == INFINITY ? "inf" : to_string(g.dist[i])) << ", Parent = " << g.parent[i] << "\n";
            }
             if (n > print_count) cout << "...\n";


            auto start_update = high_resolution_clock::now();
            updateSSSPBatch(g, changes, isInsertion);
            auto end_update = high_resolution_clock::now();
            cout << "\nSSSP batch update took "
                 << duration_cast<milliseconds>(end_update - start_update).count() << "ms" << endl;

            cout << "\nAfter batch updates:\n";
            for (int i = 1; i <= print_count; i++) {
                 cout << "Vertex " << i << ": Dist = " << (g.dist[i] == INFINITY ? "inf" : to_string(g.dist[i])) << ", Parent = " << g.parent[i] << "\n";
            }
             if (n > print_count) cout << "...\n";


        } catch (const exception& e) {
            cerr << "Error during graph processing or update: " << e.what() << endl;
            // Continue to print total time if possible, but return error code
             auto end_total = high_resolution_clock::now();
             cout << "\nTotal execution time (encountered error): "
                  << duration_cast<milliseconds>(end_total - start_total).count() << "ms" << endl;
            return 1;
        }

        // Calculate and print total execution time
        auto end_total = high_resolution_clock::now();
        cout << "\nTotal execution time: "
             << duration_cast<milliseconds>(end_total - start_total).count() << "ms" << endl;

    } catch (const exception& e) {
        cerr << "Fatal error in main: " << e.what() << endl;
        return 1;
    }
    return 0;
}