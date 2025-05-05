#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <chrono>
#include <atomic>

using namespace std;
using namespace std::chrono;

const double INFINITY_VAL = numeric_limits<double>::infinity();

// Edge structure for updates
struct Edge {
    int u, v;
    double weight;
};

// Custom atomic min operation for use in OpenMP
inline bool atomic_min(double* addr, double val) {
    double old;
    bool changed = false;
    #pragma omp critical
    {
        old = *addr;
        if (val < old) {
            *addr = val;
            changed = true;
        }
    }
    return changed;
}

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
    
    // Track vertices that need recalculation
    vector<char> need_update; // Using char for better cache alignment

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
            need_update.resize(n + 1, 0);

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

    // Build children list based on parent array - now parallel
    void buildChildren() {
        for (int i = 1; i <= n; i++) {
            children[i].clear();
        }
        
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

// Mark descendants as affected (DFS) - modified for parallelism
void markDescendants(Graph& g, int v, vector<int>& affected_vertices) {
    vector<int> stack;
    stack.push_back(v);
    
    while (!stack.empty()) {
        int current = stack.back();
        stack.pop_back();
        
        g.dist[current] = INFINITY_VAL;
        g.parent[current] = -1;
        g.need_update[current] = 1;
        affected_vertices.push_back(current);
        
        // Add all children to stack
        for (int child : g.children[current]) {
            stack.push_back(child);
        }
    }
}

// Parallel delta-stepping SSSP algorithm
void deltaStepSSSP(Graph& g, vector<int>& affected_vertices, double delta = 1.0) {
    const int num_buckets = 1000;  // Adjust based on graph diameter
    vector<vector<int>> buckets(num_buckets);
    vector<char> in_bucket(g.n + 1, 0);
    
    // Initialize buckets with affected vertices
    for (int v : affected_vertices) {
        g.need_update[v] = 0;  // Reset need_update flag
        if (g.dist[v] != INFINITY_VAL) {
            int bucket_idx = min(int(g.dist[v] / delta), num_buckets - 1);
            buckets[bucket_idx].push_back(v);
            in_bucket[v] = 1;
        }
    }
    
    // Process buckets in order
    for (int b = 0; b < num_buckets; b++) {
        while (!buckets[b].empty()) {
            vector<int> current_bucket;
            swap(current_bucket, buckets[b]);
            
            // Process current bucket in parallel
            #pragma omp parallel
            {
                vector<pair<int, double>> local_updates;
                
                #pragma omp for nowait
                for (size_t i = 0; i < current_bucket.size(); i++) {
                    int u = current_bucket[i];
                    in_bucket[u] = 0;
                    double dist_u = g.dist[u];
                    
                    // Process all outgoing edges
                    for (int j = g.row_ptr[u]; j < g.row_ptr[u + 1]; j++) {
                        int v = g.adj[j];
                        double weight = g.weights[j];
                        double new_dist = dist_u + weight;
                        
                        if (new_dist < g.dist[v]) {
                            local_updates.push_back({v, new_dist});
                        }
                    }
                }
                
                // Apply updates with proper synchronization
                #pragma omp critical
                {
                    for (auto& update : local_updates) {
                        int v = update.first;
                        double new_dist = update.second;
                        
                        if (new_dist < g.dist[v]) {
                            g.dist[v] = new_dist;
                            
                            // Find parent
                            for (int j = g.row_ptr[v]; j < g.row_ptr[v + 1]; j++) {
                                int u = g.adj[j];
                                if (abs(g.dist[v] - g.dist[u] - g.weights[j]) < 1e-6) {
                                    g.parent[v] = u;
                                    break;
                                }
                            }
                            
                            if (!in_bucket[v]) {
                                int bucket_idx = min(int(new_dist / delta), num_buckets - 1);
                                buckets[bucket_idx].push_back(v);
                                in_bucket[v] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Update SSSP for multiple edge changes
void updateSSSPBatch(Graph& g, const vector<Edge>& changes, const vector<bool>& isInsertion) {
    // Build children list for deletion handling
    auto start = high_resolution_clock::now();
    g.buildChildren();
    auto end = high_resolution_clock::now();
    cout << "Build children took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;
    
    vector<int> affected_vertices;
    
    // Step 1: Process deletions first - identify affected vertices
    start = high_resolution_clock::now();
    for (size_t i = 0; i < changes.size(); i++) {
        if (!isInsertion[i]) {
            int u = changes[i].u, v = changes[i].v;
            
            // Find both path directions
            int x = (g.dist[u] > g.dist[v]) ? u : v;
            int y = (x == u) ? v : u;
            
            if (g.parent[x] == y) { // Edge is in SSSP tree
                markDescendants(g, x, affected_vertices);
            }
        }
    }
    end = high_resolution_clock::now();
    cout << "Process deletions took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;
    
    // Step 2: Process insertions - update distances
    start = high_resolution_clock::now();
    for (size_t i = 0; i < changes.size(); i++) {
        if (isInsertion[i]) {
            int u = changes[i].u, v = changes[i].v;
            double w = changes[i].weight;
            
            // Update edge weights in CSR
            bool found_forward = false, found_backward = false;
            
            #pragma omp parallel for
            for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                if (g.adj[i] == v) {
                    g.weights[i] = w;
                    #pragma omp atomic write
                    found_forward = true;
                }
            }
            
            #pragma omp parallel for
            for (int i = g.row_ptr[v]; i < g.row_ptr[v + 1]; i++) {
                if (g.adj[i] == u) {
                    g.weights[i] = w;
                    #pragma omp atomic write
                    found_backward = true;
                }
            }
            
            // If edge doesn't exist in CSR, we'd need to add it
            // For simplicity, assuming edges already exist
            
            // Check if we can improve distances
            if (g.dist[u] + w < g.dist[v]) {
                g.dist[v] = g.dist[u] + w;
                g.parent[v] = u;
                g.need_update[v] = 1;
                affected_vertices.push_back(v);
            }
            
            if (g.dist[v] + w < g.dist[u]) {
                g.dist[u] = g.dist[v] + w;
                g.parent[u] = v;
                g.need_update[u] = 1;
                affected_vertices.push_back(u);
            }
        }
    }
    end = high_resolution_clock::now();
    cout << "Process insertions took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;
    
    // Step 3: Update affected subgraph
    start = high_resolution_clock::now();
    deltaStepSSSP(g, affected_vertices);
    end = high_resolution_clock::now();
    cout << "Delta-stepping took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;
}

// Parallel version of Dijkstra's algorithm
void dijkstra(Graph& g, int source) {
    try {
        if (source < 1 || source > g.n) {
            throw runtime_error("Invalid source vertex");
        }

        #pragma omp parallel for
        for (int i = 1; i <= g.n; i++) {
            g.dist[i] = INFINITY_VAL;
            g.parent[i] = -1;
        }
        
        g.dist[source] = 0.0;
        
        // Use a sequential priority queue but parallelize edge relaxation
        using pq_element = pair<double, int>;
        priority_queue<pq_element, vector<pq_element>, greater<pq_element>> pq;
        vector<bool> finished(g.n + 1, false);
        
        pq.push({0.0, source});
        
        while (!pq.empty()) {
            double d = pq.top().first;
            int u = pq.top().second;
            pq.pop();
            
            if (finished[u]) continue;
            finished[u] = true;
            
            // Relaxation phase can be parallelized
            vector<pq_element> next_vertices;
            
            #pragma omp parallel
            {
                vector<pq_element> thread_next;
                
                #pragma omp for nowait
                for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                    int v = g.adj[i];
                    if (finished[v]) continue;
                    
                    double w = g.weights[i];
                    double new_dist = g.dist[u] + w;
                    
                    if (new_dist < g.dist[v]) {
                        // Use atomic operations for updating distance and parent
                        if (atomic_min(&g.dist[v], new_dist)) {
                            #pragma omp critical
                            {
                                g.parent[v] = u;
                            }
                            thread_next.push_back({new_dist, v});
                        }
                    }
                }
                
                #pragma omp critical
                {
                    next_vertices.insert(next_vertices.end(), thread_next.begin(), thread_next.end());
                }
            }
            
            // Add vertices to the queue outside parallel region
            for (auto& item : next_vertices) {
                pq.push(item);
            }
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

// Reads batch updates from a file and fills changes and isInsertion vectors.
// File format: each line is either "I u v w" (insert edge u-v with weight w) or "D u v w" (delete edge u-v, w is ignored or can be 0)
void load_updates(const string& filename, vector<Edge>& changes, vector<bool>& isInsertion) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Error opening updates file: " + filename);
    }
    string type;
    int u, v;
    double w;
    while (file >> type >> u >> v >> w) {
        if (type == "I" || type == "i") {
            changes.push_back({u, v, w});
            isInsertion.push_back(true);
        } else if (type == "D" || type == "d") {
            changes.push_back({u, v, w});
            isInsertion.push_back(false);
        } else {
            cerr << "Unknown update type in updates file: " << type << endl;
        }
    }
    file.close();
}

int main(int argc, char* argv[]) {
    try {
        int n, m;
        vector<int> src, dst;
        vector<double> weights;
        
        // Set number of threads based on command-line argument or default to 4
        int num_threads = 4;
        if (argc > 1) {
            num_threads = atoi(argv[1]);
        }
        omp_set_num_threads(num_threads);
        cout << "Running with " << num_threads << " threads" << endl;

        string filename = "../Dataset/sample_graph.txt";
        if (argc > 2) {
            filename = argv[2];
        }

        string updates_filename = "../Dataset/updates.txt";
        if (argc > 3) {
            updates_filename = argv[3];
        }
        
        auto start_total = high_resolution_clock::now();
        
        try {
            auto start = high_resolution_clock::now();
            load_data(filename, n, m, src, dst, weights);
            auto end = high_resolution_clock::now();
            cout << "Loaded graph with " << n << " vertices and " << m << " edges in " 
                 << duration_cast<milliseconds>(end - start).count() << "ms" << endl;
            
            // Verify data
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
            vector<Edge> changes;
            vector<bool> isInsertion;
            load_updates(updates_filename, changes, isInsertion);

            // Verify update vertices are in range
            for (const auto& edge : changes) {
                if (edge.u <= 0 || edge.u > n || edge.v <= 0 || edge.v > n) {
                    cerr << "Invalid update edge: (" << edge.u << ", " << edge.v << ")" << endl;
                    throw runtime_error("Update edge endpoints out of range");
                }
            }

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
