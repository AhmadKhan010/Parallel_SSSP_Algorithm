#include <iostream>
#include <vector>
#include <queue> 
#include <limits>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;

const double INFINITY_VAL = numeric_limits<double>::infinity();

// Edge structure for updates (used in load_updates)
struct EdgeUpdate {
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
    vector<vector<int>> children; // Children in SSSP tree (built temporarily)
    int n;                  // Number of vertices
    int m;                  // Number of edges (original count * 2)

    // Flag for iterative relaxation
    vector<char> affected;     // Affected by any change (Affected_Flag)

    Graph(int n, int m, const vector<int>& src, const vector<int>& dst, const vector<double>& weights_in) {
        try {
            if (n <= 0 || m <= 0) {
                throw runtime_error("Invalid graph size");
            }

            this->n = n;
            // Store original edge count * 2 for CSR size calculation
            this->m = m * 2;

            // Resize vectors for 1-based indexing
            dist.resize(n + 1, INFINITY_VAL);
            parent.resize(n + 1, -1);
            children.resize(n + 1); // Resized, but only built when needed
            row_ptr.resize(n + 2, 0);
            adj.resize(this->m);
            this->weights.resize(this->m);
            affected.resize(n + 1, 0);     // Initialize flags to false

            // Build CSR with 1-based indexing
            vector<int> edge_count(n + 1, 0);
            for (int i = 0; i < m; i++) {
                if (src[i] < 1 || src[i] > n || dst[i] < 1 || dst[i] > n) {
                     throw runtime_error("Edge vertex index out of bounds during CSR build count");
                }
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
             if (row_ptr[n + 1] != this->m) {
                 // This can happen if input m doesn't match actual edges
                 cerr << "Warning: CSR final edge count mismatch. Expected " << this->m << " got " << row_ptr[n+1] << endl;
                 // Adjust adj/weights size if needed, though this indicates an input issue
                 // this->m = row_ptr[n + 1];
                 // adj.resize(this->m);
                 // this->weights.resize(this->m);
                 // For now, throw error as it suggests inconsistent input
                 throw runtime_error("CSR final edge count mismatch");
             }


            vector<int> current_pos(n + 1, 0);
            for (int i = 0; i < m; i++) {
                int u_node = src[i];
                int v_node = dst[i];
                double w = weights_in[i];

                if (u_node < 1 || u_node > n || v_node < 1 || v_node > n) {
                     throw runtime_error("Edge vertex index out of bounds during CSR population");
                }

                // Forward edge
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
        // No parallelism needed here, typically fast enough
        children.assign(n + 1, vector<int>());
        for (int i = 1; i <= n; i++) {
            if (parent[i] != -1 && parent[i] >= 1 && parent[i] <= n) {
                children[parent[i]].push_back(i);
            }
        }
    }

    // Helper to find edge index in CSR
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

    // Helper to update edge weight (or mark as deleted by setting INFINITY_VAL)
    // Note: Potential race condition if multiple threads update the same edge. Last write wins.
    void updateEdgeWeight(int u, int v, double new_weight) {
        int idx_uv = findEdgeIndex(u, v);
        if (idx_uv != -1 && idx_uv >= 0 && idx_uv < weights.size()) {
            weights[idx_uv] = new_weight;
        }
        int idx_vu = findEdgeIndex(v, u);
        if (idx_vu != -1 && idx_vu >= 0 && idx_vu < weights.size()) {
            weights[idx_vu] = new_weight;
        }
    }
};

// Mark descendants as affected (DFS) - Modifies dist, parent, and affected flags
// Called within parallel region, updates are potentially racy but rely on iterative relaxation
void markDescendants(Graph& g, int v) {
    if (v < 1 || v > g.n) return;

    // Check if already marked infinite in this pass to avoid redundant work/cycles
    // This check itself isn't perfectly thread-safe but helps prune recursion.
    if (g.dist[v] == INFINITY_VAL) return;

    g.dist[v] = INFINITY_VAL;
    g.parent[v] = -1;
    g.affected[v] = 1; // Mark as affected for relaxation phase

    // Recursively mark children based on the *current* children structure
    // This structure might be slightly outdated due to parallel updates,
    // but the relaxation phase should handle inconsistencies.
    if (v >= 1 && v < g.children.size()) {
        // Create a local copy of children to avoid issues if g.children is modified concurrently
        vector<int> current_children = g.children[v];
        for (int child : current_children) {
            markDescendants(g, child);
        }
    }
}


/**
 * Update SSSP based on Serial.cpp logic, parallelized with OpenMP
 * and using iterative relaxation instead of a priority queue for Step 3.
 */
void updateSSSP_Iterative(Graph& g, const vector<EdgeUpdate>& deletions, const vector<EdgeUpdate>& insertions) {
    auto start_total = high_resolution_clock::now();

    // --- Step 0: Preparation ---
    g.buildChildren(); // Build based on current parent state before updates

    // Reset affected flags in parallel
    #pragma omp parallel for
    for (int i = 1; i <= g.n; ++i) {
        g.affected[i] = 0;
    }

    // --- Step 1: Process Deletions (Parallel) ---
    auto start_del = high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < deletions.size(); ++i) {
        int u = deletions[i].u;
        int v = deletions[i].v;

        if (u < 1 || u > g.n || v < 1 || v > g.n) continue;

        // Identify if edge was in the SSSP tree *before* parallel updates started
        // Reading g.parent here might race with other deletions, but we proceed.
        int x = -1; // The child vertex if edge was in tree
        if (g.parent[v] == u) {
            x = v;
        } else if (g.parent[u] == v) {
            x = u;
        }

        // If it was in the tree, mark descendants (recursive, potential races)
        if (x != -1) {
            markDescendants(g, x); // Modifies dist, parent, affected
        }

        // Mark edge as deleted in CSR (potential races, but setting inf is ok)
        g.updateEdgeWeight(u, v, INFINITY_VAL);
    }
    auto end_del = high_resolution_clock::now();
    cout << "Step 1 (Process Deletions) took "
         << duration_cast<milliseconds>(end_del - start_del).count() << "ms" << endl;


    // --- Step 2: Process Insertions (Parallel) ---
    auto start_ins = high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < insertions.size(); ++i) {
        int u = insertions[i].u;
        int v = insertions[i].v;
        double w = insertions[i].weight;

        if (u < 1 || u > g.n || v < 1 || v > g.n || w < 0) continue;

        // Update edge weight (potential races, last write wins)
        g.updateEdgeWeight(u, v, w);

        // Check for path improvements (potential races on dist/parent/affected)
        // Read distances - might be stale due to concurrent updates
        double dist_u = g.dist[u];
        double dist_v = g.dist[v];

        // Check u -> v
        if (dist_u != INFINITY_VAL && dist_u + w < dist_v) {
            // Non-atomic updates - rely on iterative relaxation
            g.dist[v] = dist_u + w;
            g.parent[v] = u;
            g.affected[v] = 1;
        }
        // Check v -> u (read dist_v again in case it was just updated)
        // Re-read dist_u as well for symmetry, though less likely to change here
        dist_u = g.dist[u]; // Re-read
        dist_v = g.dist[v]; // Re-read
        if (dist_v != INFINITY_VAL && dist_v + w < dist_u) {
            // Non-atomic updates - rely on iterative relaxation
            g.dist[u] = dist_v + w;
            g.parent[u] = v;
            g.affected[u] = 1;
        }
    }
    auto end_ins = high_resolution_clock::now();
    cout << "Step 2 (Process Insertions) took "
         << duration_cast<milliseconds>(end_ins - start_ins).count() << "ms" << endl;


    // --- Step 3: Iterative Relaxation (Parallel) ---
    auto start_relax = high_resolution_clock::now();
    bool changed_in_iteration = true;
    int iteration_count = 0;

    while (changed_in_iteration) {
        iteration_count++;
        changed_in_iteration = false;
        // Create a snapshot of affected flags for this iteration
        vector<char> affected_snapshot = g.affected;

        #pragma omp parallel for schedule(dynamic) reduction(||:changed_in_iteration)
        for (int v = 1; v <= g.n; ++v) {
            if (affected_snapshot[v]) { // Process only vertices marked in snapshot
                g.affected[v] = 0; // Reset flag for the *next* iteration

                double dist_v = g.dist[v]; // Read current distance once

                if (v + 1 >= g.row_ptr.size()) continue; // Bounds check

                // Relax neighbors n of v
                for (int i = g.row_ptr[v]; i < g.row_ptr[v + 1]; ++i) {
                    if (i < 0 || i >= g.adj.size()) continue; // Bounds check

                    int n_node = g.adj[i];
                    if (n_node == -1) continue; // Skip deleted edges marked in CSR

                    double weight_vn = g.weights[i];
                    if (weight_vn == INFINITY_VAL) continue; // Skip deleted edges

                    if (n_node < 1 || n_node > g.n) continue; // Bounds check

                    // --- Relaxation Check v -> n ---
                    if (dist_v != INFINITY_VAL) {
                        double potential_dist_n = dist_v + weight_vn;
                        // Read g.dist[n_node] within the check for atomicity (though not truly atomic)
                        if (potential_dist_n < g.dist[n_node] - 1e-9) { // Epsilon comparison
                            // Non-atomic update
                            g.dist[n_node] = potential_dist_n;
                            g.parent[n_node] = v;
                            g.affected[n_node] = 1; // Mark neighbor for next iteration
                            changed_in_iteration = true;
                        }
                    }

                    // --- Relaxation Check n -> v ---
                    // Read g.dist[n_node] again for this check
                    double dist_n = g.dist[n_node];
                    if (dist_n != INFINITY_VAL) {
                        double potential_dist_v = dist_n + weight_vn;
                        // Compare with dist_v read at the start of the outer loop for v
                        if (potential_dist_v < dist_v - 1e-9) { // Epsilon comparison
                             // Non-atomic update
                            g.dist[v] = potential_dist_v;
                            g.parent[v] = n_node;
                            g.affected[v] = 1; // Mark self for next iteration
                            changed_in_iteration = true;
                            // Update local dist_v copy for subsequent checks in *this* inner loop
                            dist_v = potential_dist_v;
                        }
                    }
                } // End neighbor loop
            } // End if(affected_snapshot[v])
        } // End parallel for loop

        // Implicit barrier here
        // Reduction updates changed_in_iteration

    } // End while loop

    auto end_relax = high_resolution_clock::now();
    cout << "Step 3 (Iterative Relaxation) took "
         << duration_cast<milliseconds>(end_relax - start_relax).count()
         << "ms in " << iteration_count << " iterations" << endl;

    auto end_total = high_resolution_clock::now();
    cout << "Total SSSP Update took "
         << duration_cast<milliseconds>(end_total - start_total).count() << "ms" << endl;
}


// Sequential Dijkstra's algorithm for initial SSSP computation
// (Identical to the one in the previous Parallel_Simple.cpp)
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

// Function to load graph data from a file
// (Identical to the one in the previous Parallel_Simple.cpp)
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

// Load updates and separate them into insertions and deletions
// (Modified slightly to match EdgeUpdate struct)
void load_updates(const string& filename, vector<EdgeUpdate>& deletions, vector<EdgeUpdate>& insertions) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Error opening updates file: " + filename);
    }

    int num_updates;
    if (!(file >> num_updates)) {
        file.close();
        throw runtime_error("Error reading number of updates from file: " + filename);
    }

    string type;
    int u, v;
    double w;
    int count = 0;
    while (count < num_updates && (file >> u >> v >> w >> type)) {
        count++;
        if (u <= 0 || v <= 0) {
            cerr << "Warning: Invalid vertex ID (<=0) in updates file line " << count << ": " << u << ", " << v << endl;
            continue;
        }

        if (type == "1" || type == "i") {
            insertions.push_back({u, v, w});
        } else if (type == "0" || type == "d") {
            deletions.push_back({u, v, w});
        } else {
            cerr << "Warning: Unknown update type '" << type << "' in updates file line " << count << endl;
        }
    }
    if (count < num_updates) {
        cerr << "Warning: Expected " << num_updates << " updates, but only processed " << count << " lines." << endl;
    }
    file.close();

    cout << "Loaded " << deletions.size() << " deletions and "
         << insertions.size() << " insertions" << endl;
}

int main(int argc, char* argv[]) {
    try {
        int n, m;
        vector<int> src, dst;
        vector<double> weights;

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
            // Load the graph
            auto start = high_resolution_clock::now();
            load_data(filename, n, m, src, dst, weights);
            auto end = high_resolution_clock::now();
            cout << "Loaded graph with " << n << " vertices and " << m << " edges in "
                 << duration_cast<milliseconds>(end - start).count() << "ms" << endl;

            // Construct graph
            start = high_resolution_clock::now();
            Graph g(n, m, src, dst, weights);
            end = high_resolution_clock::now();
            cout << "Graph construction took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;

            // Compute initial SSSP with Dijkstra
            start = high_resolution_clock::now();
            dijkstra(g, 1); // Source node 1
            end = high_resolution_clock::now();
            cout << "Initial Dijkstra computation took " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;

            // Load all updates
            vector<EdgeUpdate> all_deletions, all_insertions;
            load_updates(updates_filename, all_deletions, all_insertions);

            // Basic validation of updates
            bool valid_updates = true;
            for(const auto& edge : all_deletions) {
                if (edge.u <= 0 || edge.u > n || edge.v <= 0 || edge.v > n) valid_updates = false;
            }
            for(const auto& edge : all_insertions) {
                if (edge.u <= 0 || edge.u > n || edge.v <= 0 || edge.v > n) valid_updates = false;
            }
            if (!valid_updates) {
                throw runtime_error("Update edge endpoints out of range");
            }

            // Print initial SSSP distances
            cout << "\nBefore updates:\n";
            int print_count = min(10, n);
            for (int i = 1; i <= print_count; i++) {
                if (i >= 0 && i < g.dist.size() && i < g.parent.size()) {
                    cout << "Vertex " << i << ": Dist = " << (g.dist[i] == INFINITY_VAL ? "inf" : to_string(g.dist[i]))
                         << ", Parent = " << g.parent[i] << "\n";
                }
            }
            if (n > print_count) {
                cout << "..." << endl;
            }

            // Process all updates at once
            cout << "\n===== Processing All Updates =====\n" << endl;

            // Execute the parallel update function
            start = high_resolution_clock::now();
            updateSSSP_Iterative(g, all_deletions, all_insertions);
            end = high_resolution_clock::now();
            cout << "SSSP update processing took "
                 << duration_cast<milliseconds>(end - start).count() << "ms" << endl;

            // Print updated SSSP distances after processing all updates
            cout << "\nAfter updates:\n";
            for (int i = 1; i <= print_count; i++) {
                if (i >= 0 && i < g.dist.size() && i < g.parent.size()) {
                    cout << "Vertex " << i << ": Dist = "
                         << (g.dist[i] == INFINITY_VAL ? "inf" : to_string(g.dist[i]))
                         << ", Parent = " << g.parent[i] << "\n";
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
        cout << "\nTotal execution time: " << duration_cast<milliseconds>(end_total - start_total).count() << "ms" << endl;

    } catch (const exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return 1;
    }
    return 0;
}