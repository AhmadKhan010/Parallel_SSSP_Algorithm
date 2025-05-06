#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <chrono>

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
    int n;                  // Number of vertices
    int m;                  // Number of edges (original count * 2)

    // Flags based on paper's description
    vector<char> affected_del; // Affected by deletion propagation
    vector<char> affected;     // Affected by any change (for relaxation)

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
            row_ptr.resize(n + 2, 0);
            adj.resize(this->m);
            this->weights.resize(this->m);
            affected_del.resize(n + 1, 0); // Initialize flags to false
            affected.resize(n + 1, 0);     // Initialize flags to false

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
                    throw runtime_error("CSR index calculation potentially out of bounds");
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

    // Helper to find edge index in CSR (needed for deletion marking)
    // Returns -1 if not found
    int findEdgeIndex(int u, int target_v) {
        if (u < 1 || u > n) return -1;
        if (u + 1 >= row_ptr.size()) return -1;
        for (int i = row_ptr[u]; i < row_ptr[u + 1]; ++i) {
            if (i < 0 || i >= adj.size()) continue;
            if (adj[i] == target_v) {
                return i;
            }
        }
        return -1;
    }
};


void updateSSSPBatch(Graph& g, const vector<Edge>& changes, const vector<bool>& isInsertion) {
    // Store initial parent array to represent the tree T before updates
    vector<int> initial_parent = g.parent;

    // --- Algorithm 2: Identify Affected Vertices ---
    auto start_alg2 = high_resolution_clock::now();

    // Reset flags 
    #pragma omp parallel for
    for (int i = 1; i <= g.n; ++i) {
        g.affected_del[i] = 0;
        g.affected[i] = 0;
    }

    // Process Deletions (Parallel Loop 1)
    #pragma omp parallel for
    for (size_t i = 0; i < changes.size(); ++i) {
        if (!isInsertion[i]) {
            int u = changes[i].u;
            int v = changes[i].v;
            if (u < 1 || u > g.n || v < 1 || v > g.n) continue;

            // Check if edge was in the initial SSSP tree T
            int child_node = -1;
            if (v >= 0 && v < initial_parent.size() && initial_parent[v] == u) {
                child_node = v;
            } else if (u >= 0 && u < initial_parent.size() && initial_parent[u] == v) {
                child_node = u;
            }

            if (child_node != -1) {
                if (child_node >= 1 && child_node <= g.n) {
                    g.dist[child_node] = INFINITY_VAL;
                    g.affected_del[child_node] = 1;
                    g.affected[child_node] = 1;
                }
            }

            // Mark edge as deleted in CSR (find both directions)
            int idx_uv = g.findEdgeIndex(u, v);
            if (idx_uv != -1) {
                if (idx_uv >= 0 && idx_uv < g.adj.size()) {
                    g.adj[idx_uv] = -1;
                    g.weights[idx_uv] = INFINITY_VAL;
                }
            }
            int idx_vu = g.findEdgeIndex(v, u);
            if (idx_vu != -1) {
                if (idx_vu >= 0 && idx_vu < g.adj.size()) {
                    g.adj[idx_vu] = -1;
                    g.weights[idx_vu] = INFINITY_VAL;
                }
            }
        }
    }

    // Process Insertions (Parallel Loop 2)
    #pragma omp parallel for
    for (size_t i = 0; i < changes.size(); ++i) {
        if (isInsertion[i]) {
            int u = changes[i].u;
            int v = changes[i].v;
            double w = changes[i].weight;
            if (u < 1 || u > g.n || v < 1 || v > g.n || w < 0) continue;

            // Update edge weight in CSR
            int idx_uv = g.findEdgeIndex(u, v);
            if (idx_uv != -1 && idx_uv >= 0 && idx_uv < g.weights.size() && g.adj[idx_uv] != -1) {
                g.weights[idx_uv] = w;
            }
            int idx_vu = g.findEdgeIndex(v, u);
            if (idx_vu != -1 && idx_vu >= 0 && idx_vu < g.weights.size() && g.adj[idx_vu] != -1) {
                g.weights[idx_vu] = w;
            }

            // Check potential path improvements
            double dist_u = g.dist[u];
            double dist_v = g.dist[v];

            // Check u -> v
            if (dist_u != INFINITY_VAL && dist_u + w < dist_v) {
                g.dist[v] = dist_u + w;
                g.parent[v] = u;
                g.affected[v] = 1;
            }
            // Check v -> u
            if (dist_v != INFINITY_VAL && dist_v + w < dist_u) {
                g.dist[u] = dist_v + w;
                g.parent[u] = v;
                g.affected[u] = 1;
            }
        }
    }
    auto end_alg2 = high_resolution_clock::now();
    cout << "Algorithm 2 (Identify Affected) took " << duration_cast<milliseconds>(end_alg2 - start_alg2).count() << "ms" << endl;

    // --- Algorithm 3: Update Affected Subgraph ---
    auto start_alg3 = high_resolution_clock::now();

    // Precompute initial children list for efficiency in Part 1
    vector<vector<int>> initial_children(g.n + 1);
    for (int i = 1; i <= g.n; ++i) {
        if (initial_parent[i] != -1 && initial_parent[i] >= 1 && initial_parent[i] <= g.n) {
            initial_children[initial_parent[i]].push_back(i);
        }
    }

    // Part 1: Propagate Deletions
    auto start_alg3_p1 = high_resolution_clock::now();
    bool deletion_propagated = true;
    while (deletion_propagated) {
        deletion_propagated = false;
        vector<char> affected_del_snapshot = g.affected_del;

        #pragma omp parallel for reduction(||:deletion_propagated) schedule(dynamic)
        for (int v = 1; v <= g.n; ++v) {
            if (affected_del_snapshot[v]) {
                g.affected_del[v] = 0;

                if (v >= 0 && v < initial_children.size()) {
                    for (int c : initial_children[v]) {
                        if (c >= 1 && c <= g.n) {
                            if (g.dist[c] != INFINITY_VAL) {
                                g.dist[c] = INFINITY_VAL;
                                g.parent[c] = -1;
                                g.affected_del[c] = 1;
                                g.affected[c] = 1;
                                deletion_propagated = true;
                            }
                        }
                    }
                }
            }
        }
    }
    auto end_alg3_p1 = high_resolution_clock::now();
    cout << "Algorithm 3 Part 1 (Deletion Propagate) took " << duration_cast<milliseconds>(end_alg3_p1 - start_alg3_p1).count() << "ms" << endl;

    // Part 2: Iterative Relaxation
    auto start_alg3_p2 = high_resolution_clock::now();
    bool changed_in_iteration = true;
    int iteration_count = 0;
    while (changed_in_iteration) {
        iteration_count++;
        changed_in_iteration = false;
        vector<char> affected_snapshot = g.affected;

        #pragma omp parallel for reduction(||:changed_in_iteration) schedule(dynamic)
        for (int v = 1; v <= g.n; ++v) {
            if (affected_snapshot[v]) {
                g.affected[v] = 0;

                double dist_v = g.dist[v];

                if (v + 1 >= g.row_ptr.size()) continue;
                for (int i = g.row_ptr[v]; i < g.row_ptr[v + 1]; ++i) {
                    if (i < 0 || i >= g.adj.size()) continue;

                    int n_node = g.adj[i];
                    if (n_node == -1) continue;

                    double w = g.weights[i];
                    if (w == INFINITY_VAL) continue;

                    if (n_node < 1 || n_node > g.n) continue;

                    double dist_n = g.dist[n_node];

                    if (dist_v != INFINITY_VAL) {
                        double potential_dist_n = dist_v + w;
                        if (potential_dist_n < dist_n - 1e-9) {
                            g.dist[n_node] = potential_dist_n;
                            g.parent[n_node] = v;
                            g.affected[n_node] = 1;
                            changed_in_iteration = true;
                        }
                    }
                }
            }
        }
    }
    auto end_alg3_p2 = high_resolution_clock::now();
    cout << "Algorithm 3 Part 2 (Relaxation) took " << duration_cast<milliseconds>(end_alg3_p2 - start_alg3_p2).count() << "ms"
         << " in " << iteration_count << " iterations" << endl;

    auto end_alg3 = high_resolution_clock::now();
    cout << "Algorithm 3 (Update Affected) took " << duration_cast<milliseconds>(end_alg3 - start_alg3).count() << "ms" << endl;
}

// Sequential version of Dijkstra's algorithm
void dijkstra(Graph& g, int source) {
    try {
        if (source < 1 || source > g.n) {
            throw runtime_error("Invalid source vertex");
        }

        for (int i = 1; i <= g.n; i++) {
            g.dist[i] = INFINITY_VAL;
            g.parent[i] = -1;
        }

        g.dist[source] = 0.0;

        using pq_element = pair<double, int>;
        priority_queue<pq_element, vector<pq_element>, greater<pq_element>> pq;

        pq.push({0.0, source});

        while (!pq.empty()) {
            double d = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (d > g.dist[u] + 1e-9) {
                continue;
            }

            if (u + 1 >= g.row_ptr.size()) continue;
            for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                if (i < 0 || i >= g.adj.size()) continue;

                int v = g.adj[i];
                double w = g.weights[i];

                if (v < 1 || v > g.n || w < 0 || w == INFINITY_VAL) continue;

                if (g.dist[u] != INFINITY_VAL) {
                    double new_dist_v = g.dist[u] + w;
                    if (new_dist_v < g.dist[v] - 1e-9) {
                        g.dist[v] = new_dist_v;
                        g.parent[v] = u;
                        pq.push({new_dist_v, v});
                    }
                }
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

    int max_vertex_id = 0;

    for (int i = 0; i < m; i++) {
        file >> src[i] >> dst[i] >> weights[i];
        max_vertex_id = max(max_vertex_id, max(src[i], dst[i]));
    }

    if (max_vertex_id > n) {
        cerr << "Warning: Vertices numbered up to " << max_vertex_id 
             << " but graph size is " << n << ". Updating n." << endl;
        n = max_vertex_id;
    }

    file.close();
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
        if((u <=10 || v<=10) && type == "0") {
           cout<< " Deleting edge (" << u << ", " << v << ") with weight " << w << endl;
        }

        if((u<=10 || v<=10) && type == "1") {
           cout<< " Inserting edge (" << u << ", " << v << ") with weight " << w << endl;
        }

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

int main(int argc, char* argv[]) {
    try {
        int n, m;
        vector<int> src, dst;
        vector<double> weights;

        int num_threads = 2;
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

            start = high_resolution_clock::now();
            Graph g(n, m, src, dst, weights);
            end = high_resolution_clock::now();
            cout << "Graph construction took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;

            start = high_resolution_clock::now();
            dijkstra(g, 1);
            end = high_resolution_clock::now();
            cout << "Initial Dijkstra computation took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;

            vector<Edge> changes;
            vector<bool> isInsertion;
            load_updates(updates_filename, changes, isInsertion);

            bool valid_updates = true;
            for (const auto& edge : changes) {
                if (edge.u <= 0 || edge.u > n || edge.v <= 0 || edge.v > n) {
                    cerr << "Invalid update edge: (" << edge.u << ", " << edge.v << ")" << endl;
                    valid_updates = false;
                    break;
                }
            }
            if (!valid_updates) {
                throw runtime_error("Update edge endpoints out of range");
            }

            cout << "Before updates:\n";
            int print_count = min(10, n);
            for (int i = 1; i <= print_count; i++) {
                if (i >= 0 && i < g.dist.size() && i < g.parent.size()) {
                    cout << "Vertex " << i << ": Dist = " << (g.dist[i] == INFINITY_VAL ? "inf" : to_string(g.dist[i])) << ", Parent = " << g.parent[i] << "\n";
                }
            }
            if (n > print_count) {
                cout << "..." << endl;
            }

            start = high_resolution_clock::now();
            updateSSSPBatch(g, changes, isInsertion);
            end = high_resolution_clock::now();
            cout << "\nSSSP update took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;

            cout << "\nAfter updates:\n";
            
            for (int i = 1; i <= print_count; i++) {
                if (i >= 0 && i < g.dist.size() && i < g.parent.size()) {
                    cout << "Vertex " << i << ": Dist = " << (g.dist[i] == INFINITY_VAL ? "inf" : to_string(g.dist[i])) << ", Parent = " << g.parent[i] << "\n";
                }
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
