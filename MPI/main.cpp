#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <fstream>
#include <mpi.h>
#include <metis.h>

using namespace std;

const double INFINITY = numeric_limits<double>::infinity();

// Edge structure for updates
struct Edge {
    int u, v;
    double weight;
    bool isInsertion;
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

            // Count local edges
            local_m = 0;
            for (int i = 0; i < m; i++) {
                if (partition[src[i]] == rank || partition[dst[i]] == rank) local_m += 2; // Forward and reverse
            }

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

// Update SSSP for multiple edge changes
void updateSSSPBatch(Graph& g, const vector<Edge>& changes, int rank, int size) {
    priority_queue<pair<double, int>, vector<pair<double, int>>, Compare> pq;
    g.buildChildren();

    // Process deletions
    for (const auto& change : changes) {
        if (!change.isInsertion && (g.partition[change.u] == rank || g.partition[change.v] == rank)) {
            int u = change.u, v = change.v;
            int x = (g.dist[u] > g.dist[v]) ? u : v;
            int y = (x == u) ? v : u;
            if (g.parent[x] == y && g.partition[x] == rank) {
                markDescendants(g, x, pq, rank);
            }
        }
    }

    // Synchronize after deletions
    syncGhostVertices(g, rank, size);

    // Process insertions
    for (const auto& change : changes) {
        if (change.isInsertion && (g.partition[change.u] == rank || g.partition[change.v] == rank)) {
            int u = change.u, v = change.v;
            double w = change.weight;
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
                double w = g.weights[i];
                if (g.dist[n] > g.dist[z] + w) {
                    g.dist[n] = g.dist[z] + w;
                    g.parent[n] = z;
                    if (g.partition[n] == rank) {
                        pq.push({g.dist[n], n});
                        local_change = true;
                        for (int j = g.row_ptr[n]; j < g.row_ptr[n + 1]; j++) {
                            int neighbor = g.adj[j];
                            if (g.partition[neighbor] == rank) {
                                pq.push({g.dist[neighbor], neighbor});
                            }
                        }
                    }
                }
            }
        }

        // Synchronize ghost vertices
        syncGhostVertices(g, rank, size);

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

    for (int i = 0; i < m; i++) {
        file >> src[i] >> dst[i] >> weights[i];
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
    }

    MPI_Bcast(partition.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);
    return partition;
}

int main() {
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        int n, m;
        vector<int> src, dst;
        vector<double> weights;
        vector<double> initial_dist;
        vector<int> initial_parent;

        // Load graph data and compute initial SSSP on rank 0
        if (rank == 0) {
            string filename = "../Sequential/data.txt";
            load_data(filename, n, m, src, dst, weights);
            cout << "Loaded graph with " << n << " vertices and " << m << " edges" << endl;

            // Initialize full graph for Dijkstra's
            Graph g_full(n, m, src, dst, weights, rank, vector<int>(n + 1, 0));
            dijkstra(g_full, 1);
            initial_dist = g_full.dist;
            initial_parent = g_full.parent;
            cout << "Completed initial SSSP computation" << endl;
        }

        // Broadcast graph dimensions and initial SSSP tree
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            src.resize(m);
            dst.resize(m);
            weights.resize(m);
            initial_dist.resize(n + 1);
            initial_parent.resize(n + 1);
        }
        MPI_Bcast(src.data(), m, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(dst.data(), m, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(weights.data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(initial_dist.data(), n + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(initial_parent.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Partition graph with METIS
        vector<int> partition = partitionGraph(n, m, src, dst, size, rank);

        // Initialize local graph
        Graph g(n, m, src, dst, weights, rank, partition);
        g.dist = initial_dist;
        g.parent = initial_parent;

        // Broadcast edge changes
        vector<Edge> changes;
        int num_changes = 0;
        if (rank == 0) {
            changes = {
                {2, 4, 1.5, true},
                {1, 2, 1.0, false},
                {1, 4, 2.0, true}
            };
            num_changes = changes.size();
        }
        MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) changes.resize(num_changes);
        vector<int> change_u(num_changes), change_v(num_changes);
        vector<double> change_w(num_changes);
        vector<int> change_isInsertion(num_changes);
        if (rank == 0) {
            for (int i = 0; i < num_changes; i++) {
                change_u[i] = changes[i].u;
                change_v[i] = changes[i].v;
                change_w[i] = changes[i].weight;
                change_isInsertion[i] = changes[i].isInsertion ? 1 : 0;
            }
        }
        MPI_Bcast(change_u.data(), num_changes, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(change_v.data(), num_changes, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(change_w.data(), num_changes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(change_isInsertion.data(), num_changes, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            for (int i = 0; i < num_changes; i++) {
                changes[i] = {change_u[i], change_v[i], change_w[i], change_isInsertion[i] == 1};
            }
        }

        // Print initial state
        if (rank == 0) {
            cout << "Before updates:\n";
            for (int i = 1; i <= n; i++) {
                cout << "Vertex " << i << ": Dist = " << g.dist[i] << ", Parent = " << g.parent[i] << "\n";
            }
        }

        // Update SSSP
        updateSSSPBatch(g, changes, rank, size);

        // Gather final distances and parents for output
        vector<double> global_dists(n + 1);
        vector<int> global_parents(n + 1);
        MPI_Reduce(g.dist.data(), global_dists.data(), n + 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(g.parent.data(), global_parents.data(), n + 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "\nAfter updates:\n";
            for (int i = 1; i <= n; i++) {
                cout << "Vertex " << i << ": Dist = " << global_dists[i] << ", Parent = " << global_parents[i] << "\n";
            }
        }

    } catch (const exception& e) {
        cerr << "Fatal error on rank " << rank << ": " << e.what() << endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}

