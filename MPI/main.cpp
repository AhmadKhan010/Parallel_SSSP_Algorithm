#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <fstream>
#include <mpi.h>
#include <metis.h>
#include <cstring> // Needed for memcpy

using namespace std;

const double INFINITY = numeric_limits<double>::infinity();

// Edge structure for updates
struct Edge {
    int u, v;
    double weight;
    bool isInsertion;
};

// Raw structure for broadcasting edge updates efficiently
struct EdgeUpdateRaw {
    int u;
    int v;
    double weight;
    int isInsertion; // Use int (0 or 1) for easier MPI transfer
};

// Graph structure in CSR format
struct Graph {
    vector<int> adj;        // CSR adjacency list (edge endpoints)
    vector<int> row_ptr;    // CSR row pointers
    vector<double> weights; // Edge weights
    vector<double> dist;    // Distance from source
    vector<int> parent;     // Parent in SSSP tree
    vector<vector<int>> children; // Children in SSSP tree
    vector<int> local_vertices;   // Vertices owned by this process
    vector<int> ghost_vertices;   // Vertices owned by other processes
    vector<int> partition;        // Partition assignment for all vertices
    int n;                        // Total number of vertices
    int m;                        // Total number of edges (global)
    int local_m;                  // Number of local edges

    Graph(int n, int m, const vector<int>& src, const vector<int>& dst, const vector<double>& weights, int rank, const vector<int>& partition) {
        try {
            if (n <= 0 || m <= 0) throw runtime_error("Invalid graph size");
            this->n = n;
            this->m = m * 2; // Undirected edges
            this->partition = partition;

            // Identify local and ghost vertices
            for (int i = 1; i <= n; i++) {
                if (partition[i] == rank) local_vertices.push_back(i);
            }

            // Reserve space before counting
            adj.reserve(2 * m);  // More space to be safe
            this->weights.reserve(2 * m);
            
            // Count local edges - safer calculation
            local_m = 0;
            for (int i = 0; i < m; i++) {
                if (partition[src[i]] == rank) local_m++;
                if (partition[dst[i]] == rank) local_m++;
            }
            local_m = min(local_m, 2 * m); // Safety check
            
            // Resize all vectors to avoid out-of-bounds accesses
            dist.resize(n + 1, INFINITY);
            parent.resize(n + 1, -1);
            children.resize(n + 1);
            row_ptr.resize(n + 2, 0);
            adj.resize(local_m);
            this->weights.resize(local_m);

            // Build CSR for local vertices
            vector<int> edge_count(n + 1, 0);
            for (int i = 0; i < m; i++) {
                if (partition[src[i]] == rank) edge_count[src[i]]++;
                if (partition[dst[i]] == rank) edge_count[dst[i]]++;
            }

            row_ptr[1] = 0;
            for (int i = 1; i <= n; i++) {
                row_ptr[i + 1] = row_ptr[i] + edge_count[i];
                if (row_ptr[i + 1] > local_m) throw runtime_error("CSR index out of bounds");
            }

            vector<int> current_pos(n + 1, 0);
            for (int i = 0; i < m; i++) {
                if (partition[src[i]] == rank) {
                    int v = src[i];
                    int pos = row_ptr[v] + current_pos[v]++;
                    adj[pos] = dst[i];
                    this->weights[pos] = weights[i];
                    if (partition[dst[i]] != rank && find(ghost_vertices.begin(), ghost_vertices.end(), dst[i]) == ghost_vertices.end()) {
                        ghost_vertices.push_back(dst[i]);
                    }
                }
                if (partition[dst[i]] == rank) {
                    int v = dst[i];
                    int pos = row_ptr[v] + current_pos[v]++;
                    adj[pos] = src[i];
                    this->weights[pos] = weights[i];
                    if (partition[src[i]] != rank && find(ghost_vertices.begin(), ghost_vertices.end(), src[i]) == ghost_vertices.end()) {
                        ghost_vertices.push_back(src[i]);
                    }
                }
            }
        } catch (const exception& e) {
            cerr << "Error creating graph on rank " << rank << ": " << e.what() << endl;
            throw;
        }
    }

    void buildChildren() {
        children.assign(n + 1, vector<int>());
        for (int i : local_vertices) {
            if (parent[i] != -1 && partition[parent[i]] == partition[i]) {
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

// Mark descendants as affected (DFS)
void markDescendants(Graph& g, int v, priority_queue<pair<double, int>, vector<pair<double, int>>, Compare>& pq, int rank) {
    if (g.partition[v] == rank) {
        g.dist[v] = INFINITY;
        g.parent[v] = -1;
        pq.push({g.dist[v], v});
        for (int child : g.children[v]) {
            markDescendants(g, child, pq, rank);
        }
    }
}

// Synchronize ghost vertex distances
void syncGhostVertices(Graph& g, int rank, int size) {
    vector<double> send_dists(g.n + 1, INFINITY);
    for (int v : g.local_vertices) send_dists[v] = g.dist[v];

    vector<double> recv_dists((g.n + 1) * size);
    MPI_Allgather(send_dists.data(), g.n + 1, MPI_DOUBLE, recv_dists.data(), g.n + 1, MPI_DOUBLE, MPI_COMM_WORLD);

    for (int v : g.ghost_vertices) {
        double min_dist = INFINITY;
        for (int i = 0; i < size; i++) {
            min_dist = min(min_dist, recv_dists[i * (g.n + 1) + v]);
        }
        g.dist[v] = min_dist;
    }
}

// Remove an edge from the local CSR representation (both directions for undirected)
void removeEdgeCSR(Graph& g, int u, int v) {
    // Remove v from u's adjacency
    for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
        if (g.adj[i] == v) {
            g.adj[i] = -1; // Mark as removed
            g.weights[i] = INFINITY;
        }
    }
    // Remove u from v's adjacency
    for (int i = g.row_ptr[v]; i < g.row_ptr[v + 1]; ++i) {
        if (g.adj[i] == u) {
            g.adj[i] = -1;
            g.weights[i] = INFINITY;
        }
    }
}

// Update SSSP for multiple edge changes
void updateSSSPBatch(Graph& g, const vector<Edge>& changes, int rank, int size) {
    priority_queue<pair<double, int>, vector<pair<double, int>>, Compare> pq;
    g.buildChildren();

    // Process deletions
    for (const auto& change : changes) {
        if (!change.isInsertion && (g.partition[change.u] == rank || g.partition[change.v] == rank)) {
            int u = change.u, v = change.v;
            // Remove the edge from the local CSR
            if (g.partition[u] == rank) removeEdgeCSR(g, u, v);
            if (g.partition[v] == rank) removeEdgeCSR(g, v, u);

            // If the deleted edge was a parent-child in the SSSP tree, mark descendants
            int x = (g.dist[u] > g.dist[v]) ? u : v;
            int y = (x == u) ? v : u;
            if (g.parent[x] == y && g.partition[x] == rank) {
                markDescendants(g, x, pq, rank);
            }
        }
    }

    // Synchronize after deletions
    MPI_Barrier(MPI_COMM_WORLD);
    syncGhostVertices(g, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);

    // Process insertions
    for (const auto& change : changes) {
        if (change.isInsertion && (g.partition[change.u] == rank || g.partition[change.v] == rank)) {
            int u = change.u, v = change.v;
            double w = change.weight;
            // Insert edge into local CSR if not present (for robustness)
            bool found_uv = false, found_vu = false;
            for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
                if (g.adj[i] == v) { found_uv = true; g.weights[i] = w; }
            }
            for (int i = g.row_ptr[v]; i < g.row_ptr[v + 1]; ++i) {
                if (g.adj[i] == u) { found_vu = true; g.weights[i] = w; }
            }
            // If not found, skip (robustness: only update SSSP, not CSR structure)

            if (g.partition[v] == rank && g.dist[u] + w < g.dist[v]) {
                g.dist[v] = g.dist[u] + w;
                g.parent[v] = u;
                pq.push({g.dist[v], v});
            }
            if (g.partition[u] == rank && g.dist[v] + w < g.dist[u]) {
                g.dist[u] = g.dist[v] + w;
                g.parent[u] = v;
                pq.push({g.dist[u], u});
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    syncGhostVertices(g, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);

    // Update affected subgraph
    bool local_change = true;
    while (local_change) {
        local_change = false;
        while (!pq.empty()) {
            auto top = pq.top();
            double dist = top.first;
            int z = top.second;
            pq.pop();

            if (dist > g.dist[z] + 1e-9) continue;

            for (int i = g.row_ptr[z]; i < g.row_ptr[z + 1]; i++) {
                int n = g.adj[i];
                if (n == -1) continue; // Skip removed edges
                double w = g.weights[i];
                if (g.dist[n] > g.dist[z] + w) {
                    g.dist[n] = g.dist[z] + w;
                    g.parent[n] = z;
                    if (g.partition[n] == rank) {
                        pq.push({g.dist[n], n});
                        local_change = true;
                        // No need to push neighbors of n here, just n is enough
                    }
                }
            }
        }

        // Synchronize ghost vertices
        MPI_Barrier(MPI_COMM_WORLD);
        syncGhostVertices(g, rank, size);
        MPI_Barrier(MPI_COMM_WORLD);

        // Check for global convergence
        int local_flag = local_change ? 1 : 0;
        int global_flag;
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        local_change = global_flag > 0;
    }
}

void dijkstra(Graph& g, int source) {
    try {
        if (source < 1 || source > g.n) throw runtime_error("Invalid source vertex");

        fill(g.dist.begin(), g.dist.end(), INFINITY);
        fill(g.parent.begin(), g.parent.end(), -1);
        
        priority_queue<pair<double, int>, vector<pair<double, int>>, Compare> pq;
        vector<bool> visited(g.n + 1, false);
        
        g.dist[source] = 0.0;
        pq.push({0.0, source});
        
        while (!pq.empty()) {
            double d = pq.top().first;
            int u = pq.top().second;
            pq.pop();
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                int v = g.adj[i];
                if (visited[v]) continue;
                
                double w = g.weights[i];
                if (g.dist[u] + w < g.dist[v]) {
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

void load_data(string filename, int& n, int& m, vector<int>& src, vector<int>& dst, vector<double>& weights) {
    ifstream file(filename);
    if (!file.is_open()) throw runtime_error("Error opening file: " + filename);

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

vector<int> partitionGraph(int n, int m, const vector<int>& src, const vector<int>& dst, int nparts, int rank) {
    vector<int> partition(n + 1, 0);
    if (rank == 0) {
        idx_t nvtxs = n, ncon = 1, nparts_idx = nparts;
        vector<idx_t> xadj(n + 1), adjncy(m * 2);
        xadj[0] = 0;

        vector<int> edge_count(n + 1, 0);
        for (int i = 0; i < m; i++) {
            edge_count[src[i]]++;
            edge_count[dst[i]]++;
        }

        for (int i = 0; i < n; i++) {
            xadj[i + 1] = xadj[i] + edge_count[i];
        }

        vector<int> current_pos(n + 1, 0);
        for (int i = 0; i < m; i++) {
            int u = src[i], v = dst[i];
            adjncy[xadj[u - 1] + current_pos[u]++] = v - 1;
            adjncy[xadj[v - 1] + current_pos[v]++] = u - 1;
        }

        vector<idx_t> part(n);
        idx_t objval;
        int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), NULL, NULL, NULL,
                                      &nparts_idx, NULL, NULL, NULL, &objval, part.data());
        if (ret != METIS_OK) throw runtime_error("METIS partitioning failed");

        for (int i = 0; i < n; i++) partition[i + 1] = part[i];

        //Print partitioning result
        cout << "Partitioning result: ";
        
    }

    MPI_Bcast(partition.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);
    return partition;
}
void load_updates(const string& filename, vector<Edge>& changes) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Error opening updates file: " + filename);
    }
    int num_updates;
    file >> num_updates;

    int u, v;
    double w;
    int type;
    while (file >> u >> v >> w >> type) {
        if (type == 1) {
            changes.push_back({u, v, w, true});
        } else if (type == 0) {
            changes.push_back({u, v, w, false});
        } else {
            cerr << "Unknown update type in updates file: " << type << endl;
        }
    }
    file.close();
}


int main() {
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);

    try {
        int n, m;
        vector<int> src, dst;
        vector<double> weights;
        vector<double> initial_dist;
        vector<int> initial_parent;

        vector<char> graph_data_buffer;
        vector<char> sssp_data_buffer;
        long long graph_buffer_size = 0;
        long long sssp_buffer_size = 0;

        auto start_time = MPI_Wtime();

        // Load graph data and compute initial SSSP on rank 0
        if (rank == 0) {
            string filename = "../Dataset/sample_graph.txt";

            try {
                load_data(filename, n, m, src, dst, weights);
                cout << "Loaded graph with " << n << " vertices and " << m << " edges" << endl;

                // Verify data
                for (int i = 0; i < m; i++) {
                    if (src[i] <= 0 || src[i] > n || dst[i] <= 0 || dst[i] > n) {
                        cerr << "Invalid edge: (" << src[i] << ", " << dst[i] << ")" << endl;
                        throw runtime_error("Edge endpoints out of range");
                    }
                }

                // Initialize full graph for Dijkstra's
                vector<int> init_partition(n + 1, 0); // Dummy partition for full graph
                Graph g_full(n, m, src, dst, weights, rank, init_partition);
                dijkstra(g_full, 1);
                initial_dist = g_full.dist;
                initial_parent = g_full.parent;
                cout << "Completed initial SSSP computation" << endl;

                // Pack graph data for broadcast
                size_t src_bytes = m * sizeof(int);
                size_t dst_bytes = m * sizeof(int);
                size_t weights_bytes = m * sizeof(double);
                graph_buffer_size = src_bytes + dst_bytes + weights_bytes;
                graph_data_buffer.resize(graph_buffer_size);
                char* ptr = graph_data_buffer.data();
                memcpy(ptr, src.data(), src_bytes); ptr += src_bytes;
                memcpy(ptr, dst.data(), dst_bytes); ptr += dst_bytes;
                memcpy(ptr, weights.data(), weights_bytes);

                // Pack SSSP data for broadcast
                size_t dist_bytes = (n + 1) * sizeof(double);
                size_t parent_bytes = (n + 1) * sizeof(int);
                sssp_buffer_size = dist_bytes + parent_bytes;
                sssp_data_buffer.resize(sssp_buffer_size);
                ptr = sssp_data_buffer.data();
                memcpy(ptr, initial_dist.data(), dist_bytes); ptr += dist_bytes;
                memcpy(ptr, initial_parent.data(), parent_bytes);

            }
            catch (const exception& e) {
                cerr << "Error in graph loading/initialization: " << e.what() << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        // Broadcast graph dimensions
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Broadcast packed graph data size and then data
        MPI_Bcast(&graph_buffer_size, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            graph_data_buffer.resize(graph_buffer_size);
            // Resize local vectors based on received m
            src.resize(m);
            dst.resize(m);
            weights.resize(m);
        }
        MPI_Bcast(graph_data_buffer.data(), graph_buffer_size, MPI_BYTE, 0, MPI_COMM_WORLD);

        // Broadcast packed SSSP data size and then data
        MPI_Bcast(&sssp_buffer_size, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            sssp_data_buffer.resize(sssp_buffer_size);
            // Resize local vectors based on received n
            initial_dist.resize(n + 1);
            initial_parent.resize(n + 1);
        }
        MPI_Bcast(sssp_data_buffer.data(), sssp_buffer_size, MPI_BYTE, 0, MPI_COMM_WORLD);

        // Unpack data on non-zero ranks
        if (rank != 0) {
            // Unpack graph data
            size_t src_bytes = m * sizeof(int);
            size_t dst_bytes = m * sizeof(int);
            size_t weights_bytes = m * sizeof(double);
            const char* ptr = graph_data_buffer.data();
            memcpy(src.data(), ptr, src_bytes); ptr += src_bytes;
            memcpy(dst.data(), ptr, dst_bytes); ptr += dst_bytes;
            memcpy(weights.data(), ptr, weights_bytes);

            // Unpack SSSP data
            size_t dist_bytes = (n + 1) * sizeof(double);
            size_t parent_bytes = (n + 1) * sizeof(int);
            ptr = sssp_data_buffer.data();
            memcpy(initial_dist.data(), ptr, dist_bytes); ptr += dist_bytes;
            memcpy(initial_parent.data(), ptr, parent_bytes);
        }

        // Partition graph with METIS
        vector<int> partition = partitionGraph(n, m, src, dst, size, rank);

        // Initialize local graph
        Graph g(n, m, src, dst, weights, rank, partition);
        g.dist = initial_dist; // Assign broadcasted initial state
        g.parent = initial_parent; // Assign broadcasted initial state

        // Broadcast edge changes 
        vector<Edge> changes;
        vector<EdgeUpdateRaw> changes_raw;
        int num_changes = 0;

        if (rank == 0) {
            string updates_filename = "../Dataset/updates.txt";
            try {
                vector<Edge> loaded_changes;
                load_updates(updates_filename, loaded_changes);
                num_changes = loaded_changes.size();
                cout << "Loaded " << num_changes << " updates" << endl;

                changes_raw.resize(num_changes);
                for (int i = 0; i < num_changes; ++i) {
                    changes_raw[i] = {loaded_changes[i].u, loaded_changes[i].v, loaded_changes[i].weight, loaded_changes[i].isInsertion ? 1 : 0};
                }
                changes = loaded_changes; // Keep on rank 0

            } catch (const exception& e) {
                cerr << "Error loading updates: " << e.what() << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            changes_raw.resize(num_changes);
        }
        MPI_Bcast(changes_raw.data(), num_changes * sizeof(EdgeUpdateRaw), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            changes.resize(num_changes);
            for (int i = 0; i < num_changes; i++) {
                changes[i] = {changes_raw[i].u, changes_raw[i].v, changes_raw[i].weight, changes_raw[i].isInsertion == 1};
            }
        }

        int print_count = min(10, n);
        // Print initial state (using the already available initial_dist/parent)
        if (rank == 0) {
            cout << "Before updates:\n";
            for (int i = 1; i <= print_count; i++) {
                 cout << "Vertex " << i << ": Dist = " << initial_dist[i] << ", Parent = " << initial_parent[i] << "\n";
            }
             if (n > print_count) cout << "..." << endl;
        }

        // Update SSSP
        updateSSSPBatch(g, changes, rank, size);

        // Gather final distances and parents for output
        vector<double> global_dists(n + 1);
        vector<int> global_parents(n + 1);
        MPI_Reduce(g.dist.data(), global_dists.data(), n + 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(g.parent.data(), global_parents.data(), n + 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        
        auto end_time = MPI_Wtime();

        if (rank == 0) {
            cout << "Total execution time: " << (end_time - start_time) * 1000 << " ms" << endl;
        }

        if (rank == 0) {
            cout << "\nAfter updates:\n";
            for (int i = 1; i <= print_count; i++) {
                cout << "Vertex " << i << ": Dist = " << (global_dists[i] == INFINITY ? "inf" : to_string(global_dists[i]))
                     << ", Parent = " << global_parents[i] << "\n";
            }
             if (n > print_count) cout << "..." << endl;
        }

    } catch (const exception& e) {
        cerr << "Fatal error on rank " << rank << ": " << e.what() << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);  
        return 1;
    }

    MPI_Finalize();
    return 0;
}

