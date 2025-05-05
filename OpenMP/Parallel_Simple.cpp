#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

const double INFINITY_VAL = numeric_limits<double>::infinity();

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
    vector<vector<int>> children; // Children in SSSP tree
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
            dist.resize(n + 1, INFINITY_VAL);
            parent.resize(n + 1, -1);
            children.resize(n + 1);
            row_ptr.resize(n + 2, 0);
            adj.resize(this->m);
            this->weights.resize(this->m);

            // Build CSR with 1-based indexing
            vector<int> edge_count(n + 1, 0);
            for (int i = 0; i < m; i++) {
                edge_count[src[i]]++;
                edge_count[dst[i]]++;
            }

            row_ptr[1] = 0;
            for (int i = 1; i <= n; i++) {
                row_ptr[i + 1] = row_ptr[i] + edge_count[i];
                if (row_ptr[i + 1] > this->m) {
                    throw runtime_error("CSR index out of bounds");
                }
            }

            vector<int> current_pos(n + 1, 0);
            for (int i = 0; i < m; i++) {
                // Forward edge
                int v = src[i];
                int pos = row_ptr[v] + current_pos[v]++;
                adj[pos] = dst[i];
                this->weights[pos] = weights[i];
                // Reverse edge
                v = dst[i];
                pos = row_ptr[v] + current_pos[v]++;
                adj[pos] = src[i];
                this->weights[pos] = weights[i];
            }
        } catch (const exception& e) {
            cerr << "Error creating graph: " << e.what() << endl;
            throw;
        }
    }

    // Build children list based on parent array
    void buildChildren() {
        // Clear all children vectors
        for (int i = 0; i <= n; i++) {
            children[i].clear();
        }
        
        // Parallel filling of children list
        #pragma omp parallel for
        for (int i = 1; i <= n; i++) {
            if (parent[i] != -1) {
                #pragma omp critical
                {
                    children[parent[i]].push_back(i);
                }
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

// Atomic distance update - lock-free version that's much faster than critical sections
inline bool atomic_min_distance(double* addr, double val) {
    uint64_t* int_addr = reinterpret_cast<uint64_t*>(addr);
    uint64_t expected = *int_addr;
    uint64_t new_val = *reinterpret_cast<uint64_t*>(&val);
    while (val < *reinterpret_cast<double*>(&expected)) {
        if (__atomic_compare_exchange_n(
                int_addr, &expected, new_val, false,
                __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
            return true;
        }
    }
    return false;
}

// Update SSSP for multiple edge changes - high-performance parallelization
void updateSSSPBatch(Graph& g, const vector<Edge>& changes, const vector<bool>& isInsertion) {
    g.buildChildren();

    vector<char> affected(g.n + 1, 0);
    vector<char> affectedDel(g.n + 1, 0);

    // === Step 1: Process deletions in parallel (Algorithm 2, lines 4-9) ===
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < changes.size(); i++) {
        if (!isInsertion[i]) {
            int u = changes[i].u, v = changes[i].v;
            // Check if edge is in SSSP tree
            if (g.parent[u] == v || g.parent[v] == u) {
                int y = (g.dist[u] > g.dist[v]) ? u : v;
                g.dist[y] = INFINITY_VAL;
                g.parent[y] = -1;
                affectedDel[y] = 1;
                affected[y] = 1;
                // Mark edge as deleted in CSR
                for (int j = g.row_ptr[u]; j < g.row_ptr[u + 1]; j++)
                    if (g.adj[j] == v) g.weights[j] = INFINITY_VAL;
                for (int j = g.row_ptr[v]; j < g.row_ptr[v + 1]; j++)
                    if (g.adj[j] == u) g.weights[j] = INFINITY_VAL;
            }
        }
    }

    // === Step 2: Mark all descendants of affectedDel as affected and set their dist to infinity (Algorithm 3, lines 2-8) ===
    bool anyDel = true;
    while (anyDel) {
        anyDel = false;
        #pragma omp parallel for schedule(dynamic)
        for (int v = 1; v <= g.n; v++) {
            if (affectedDel[v]) {
                affectedDel[v] = 0;
                for (int c : g.children[v]) {
                    // Only mark if not already infinity (avoid cycles)
                    if (g.dist[c] != INFINITY_VAL) {
                        g.dist[c] = INFINITY_VAL;
                        g.parent[c] = -1;
                        affectedDel[c] = 1;
                        affected[c] = 1;
                        anyDel = true;
                    }
                }
            }
        }
    }

    // === Step 3: Process insertions in parallel (Algorithm 2, lines 10-19) ===
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < changes.size(); i++) {
        if (isInsertion[i]) {
            int u = changes[i].u, v = changes[i].v;
            double w = changes[i].weight;
            int x, y;
            if (g.dist[u] > g.dist[v]) {
                x = v; y = u;
            } else {
                x = u; y = v;
            }
            // Only process if y is not disconnected (avoid cycles)
            if (g.dist[x] != INFINITY_VAL && g.dist[y] > g.dist[x] + w) {
                g.dist[y] = g.dist[x] + w;
                g.parent[y] = x;
                affected[y] = 1;
                // Mark edge as inserted in CSR
                for (int j = g.row_ptr[u]; j < g.row_ptr[u + 1]; j++)
                    if (g.adj[j] == v) g.weights[j] = w;
                for (int j = g.row_ptr[v]; j < g.row_ptr[v + 1]; j++)
                    if (g.adj[j] == u) g.weights[j] = w;
            }
        }
    }

    // === Step 4: Iteratively relax affected vertices and their neighbors (Algorithm 3, lines 9-20) ===
    bool any = true;
    while (any) {
        any = false;
        #pragma omp parallel for schedule(dynamic)
        for (int v = 1; v <= g.n; v++) {
            if (affected[v]) {
                affected[v] = 0;
                for (int i = g.row_ptr[v]; i < g.row_ptr[v + 1]; i++) {
                    int n = g.adj[i];
                    double w = g.weights[i];
                    if (w == INFINITY_VAL) continue;
                    // Only relax if both vertices are not disconnected (avoid cycles)
                    if (g.dist[v] != INFINITY_VAL && g.dist[n] > g.dist[v] + w) {
                        g.dist[n] = g.dist[v] + w;
                        g.parent[n] = v;
                        affected[n] = 1;
                        any = true;
                    }
                    if (g.dist[n] != INFINITY_VAL && g.dist[v] > g.dist[n] + w) {
                        g.dist[v] = g.dist[n] + w;
                        g.parent[v] = n;
                        affected[v] = 1;
                        any = true;
                    }
                }
            }
        }
    }
}

// Parallel Dijkstra optimized for better scaling
void dijkstra(Graph& g, int source) {
    try {
        if (source < 1 || source > g.n) {
            throw runtime_error("Invalid source vertex");
        }

        // Initialize in parallel with better cache layout
        #pragma omp parallel for
        for (int i = 1; i <= g.n; i++) {
            g.dist[i] = INFINITY_VAL;
            g.parent[i] = -1;
        }
        
        g.dist[source] = 0.0;
        
        // Use atomic frontier expansion instead of priority queue
        vector<bool> visited(g.n + 1, false);
        vector<bool> in_frontier(g.n + 1, false);
        vector<int> frontier;
        frontier.push_back(source);
        in_frontier[source] = true;
        
        while (!frontier.empty()) {
            vector<int> next_frontier;
            
            // Mark current frontier as visited
            #pragma omp parallel for
            for (size_t i = 0; i < frontier.size(); i++) {
                visited[frontier[i]] = true;
                in_frontier[frontier[i]] = false;
            }
            
            // Process each vertex in the frontier in parallel
            #pragma omp parallel
            {
                vector<int> local_next;
                
                #pragma omp for schedule(dynamic, 64)
                for (size_t i = 0; i < frontier.size(); i++) {
                    int u = frontier[i];
                    double dist_u = g.dist[u];
                    
                    for (int j = g.row_ptr[u]; j < g.row_ptr[u + 1]; j++) {
                        int v = g.adj[j];
                        if (visited[v]) continue;
                        
                        double weight = g.weights[j];
                        double new_dist = dist_u + weight;
                        
                        if (new_dist < g.dist[v]) {
                            // Use atomic update to avoid race conditions
                            if (atomic_min_distance(&g.dist[v], new_dist)) {
                                // Update parent
                                g.parent[v] = u;
                                
                                // Add to next frontier if not already there
                                if (!in_frontier[v]) {
                                    in_frontier[v] = true;
                                    local_next.push_back(v);
                                }
                            }
                        }
                    }
                }
                
                // Combine local frontiers
                #pragma omp critical
                {
                    next_frontier.insert(next_frontier.end(), local_next.begin(), local_next.end());
                }
            }
            
            frontier = std::move(next_frontier);
        }
    } catch (const exception& e) {
        cerr << "Error in Dijkstra's algorithm: " << e.what() << endl;
        throw;
    }
}

void load_data(string filename, int& n, int& m, vector<int>& src, vector<int>& dst, vector<double>& weights) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Error opening file: " + filename);
    }

    file >> n >> m;
    src.resize(m);
    dst.resize(m);
    weights.resize(m);

    // Add bounds checking to ensure all vertex IDs are valid
    int max_vertex_id = 0;
    
    for (int i = 0; i < m; i++) {
        file >> src[i] >> dst[i] >> weights[i];
        max_vertex_id = max(max_vertex_id, max(src[i], dst[i]));
    }
    
    // If max vertex ID is larger than reported n, update n
    if (max_vertex_id > n) {
        cerr << "Warning: Vertices numbered up to " << max_vertex_id 
             << " but graph size is " << n << ". Updating n." << endl;
        n = max_vertex_id;
    }
    
    file.close();
}

int main(int argc, char* argv[]) {
    try {
        // Start timing the whole program
        auto start_total = high_resolution_clock::now();
        
        int n, m;
        vector<int> src, dst;
        vector<double> weights;
        
        // Set number of threads based on command-line argument or default to system max
        int num_threads = omp_get_max_threads();
        if (argc > 1) {
            num_threads = atoi(argv[1]);
        }
        omp_set_num_threads(num_threads);
        cout << "Running with " << num_threads << " threads" << endl;

        string filename = "../Dataset/bitcoin.txt";
        if (argc > 2) {
            filename = argv[2];
        }
        
        try {
            auto start = high_resolution_clock::now();
            load_data(filename, n, m, src, dst, weights);
            auto end = high_resolution_clock::now();
            cout << "Loaded graph with " << n << " vertices and " << m << " edges in " 
                 << duration_cast<milliseconds>(end - start).count() << "ms" << endl;
            
            // Verify data in parallel
            bool valid_data = true;
            #pragma omp parallel for reduction(&:valid_data)
            for (int i = 0; i < m; i++) {
                if (src[i] <= 0 || src[i] > n || dst[i] <= 0 || dst[i] > n) {
                    #pragma omp critical
                    {
                        cerr << "Invalid edge: (" << src[i] << ", " << dst[i] << ")" << endl;
                    }
                    valid_data = false;
                }
            }
            
            if (!valid_data) {
                throw runtime_error("Edge endpoints out of range");
            }
            
            // Initialize graph
            start = high_resolution_clock::now();
            Graph g(n, m, src, dst, weights);
            end = high_resolution_clock::now();
            cout << "Graph construction took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;

            // Initialize SSSP tree using Dijkstra's algorithm from vertex 1
            start = high_resolution_clock::now();
            dijkstra(g, 1);
            end = high_resolution_clock::now();
            cout << "Initial Dijkstra computation took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;

            // Example batch of updates (using 1-based vertex numbers)
            vector<Edge> changes = {
                {2, 4, 1.5},  // Insertion
                {1, 2, 1.0},  // Deletion
                {1, 4, 2.0}   // Insertion
            };
            
            // Verify update vertices are in range
            for (const auto& edge : changes) {
                if (edge.u <= 0 || edge.u > n || edge.v <= 0 || edge.v > n) {
                    cerr << "Invalid update edge: (" << edge.u << ", " << edge.v << ")" << endl;
                    throw runtime_error("Update edge endpoints out of range");
                }
            }
            
            vector<bool> isInsertion = {true, false, true};

            cout << "Before updates:\n";
            // Only print first 10 vertices for large graphs
            int print_count = min(10, n);
            for (int i = 1; i <= print_count; i++) {
                cout << "Vertex " << i << ": Dist = " << g.dist[i] << ", Parent = " << g.parent[i] << "\n";
            }
            if (n > print_count) {
                cout << "..." << endl;
            }

            start = high_resolution_clock::now();
            updateSSSPBatch(g, changes, isInsertion);
            end = high_resolution_clock::now();
            cout << "SSSP update took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;

            cout << "\nAfter updates:\n";
            for (int i = 1; i <= print_count; i++) {
                cout << "Vertex " << i << ": Dist = " << g.dist[i] << ", Parent = " << g.parent[i] << "\n";
            }
            if (n > print_count) {
                cout << "..." << endl;
            }

        } catch (const exception& e) {
            cerr << "Error during execution: " << e.what() << endl;
            return 1;
        }
        
        auto end_total = high_resolution_clock::now();
        cout << "Total execution time: " << duration_cast<milliseconds>(end_total - start_total).count() << "ms" << endl;
        
    } catch (const exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return 1;
    }
    return 0;
}
