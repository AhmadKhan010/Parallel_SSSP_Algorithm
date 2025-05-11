#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <fstream>
#include <mpi.h>
#include <omp.h>
#include <metis.h>
#include <cstring>
#include <chrono>

using namespace std;
using namespace std::chrono;

const double INFINITY_VAL = numeric_limits<double>::infinity();

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
            dist.resize(n + 1, INFINITY_VAL);
            parent.resize(n + 1, -1);
            children.resize(n + 1);
            row_ptr.resize(n + 2, 0);
            adj.resize(local_m);
            this->weights.resize(local_m);

            // Build CSR for local vertices using OpenMP for parallelism
            vector<int> edge_count(n + 1, 0);
            
            #pragma omp parallel for reduction(+:edge_count[:n+1])
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
            vector<vector<int>> thread_ghost_vertices;
            
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                int num_threads = omp_get_num_threads();
                
                if (thread_id == 0) {
                    thread_ghost_vertices.resize(num_threads);
                }
                
                #pragma omp barrier
                
                vector<int>& local_ghost = thread_ghost_vertices[thread_id];
                
                #pragma omp for
                for (int i = 0; i < m; i++) {
                    if (partition[src[i]] == rank) {
                        int v = src[i];
                        int pos = -1;
                        
                        #pragma omp critical
                        {
                            pos = row_ptr[v] + current_pos[v]++;
                        }
                        
                        if (pos >= 0 && pos < local_m) {
                            adj[pos] = dst[i];
                            this->weights[pos] = weights[i];
                            
                            // Track ghost vertices
                            if (partition[dst[i]] != rank) {
                                local_ghost.push_back(dst[i]);
                            }
                        }
                    }
                    
                    if (partition[dst[i]] == rank) {
                        int v = dst[i];
                        int pos = -1;
                        
                        #pragma omp critical
                        {
                            pos = row_ptr[v] + current_pos[v]++;
                        }
                        
                        if (pos >= 0 && pos < local_m) {
                            adj[pos] = src[i];
                            this->weights[pos] = weights[i];
                            
                            // Track ghost vertices
                            if (partition[src[i]] != rank) {
                                local_ghost.push_back(src[i]);
                            }
                        }
                    }
                }
            }
            
            // Merge thread-local ghost vertices
            for (auto& thread_ghosts : thread_ghost_vertices) {
                for (int v : thread_ghosts) {
                    if (find(ghost_vertices.begin(), ghost_vertices.end(), v) == ghost_vertices.end()) {
                        ghost_vertices.push_back(v);
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
        
        #pragma omp parallel for
        for (size_t idx = 0; idx < local_vertices.size(); idx++) {
            int i = local_vertices[idx];
            if (parent[i] != -1 && partition[parent[i]] == partition[i]) {
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

// Mark descendants as affected (DFS) - OpenMP parallel version
void markDescendants(Graph& g, int v, vector<int>& affected_vertices, int rank) {
    if (g.partition[v] == rank) {
        vector<int> stack;
        stack.push_back(v);
        
        while (!stack.empty()) {
            int current = stack.back();
            stack.pop_back();
            
            // Skip if not local
            if (g.partition[current] != rank) continue;
            
            g.dist[current] = INFINITY_VAL;
            g.parent[current] = -1;
            affected_vertices.push_back(current);
            
            // Add children to stack
            for (int child : g.children[current]) {
                stack.push_back(child);
            }
        }
    }
}

// Thread-safe priority queue wrapper for OpenMP
class ThreadSafePriorityQueue {
private:
    priority_queue<pair<double, int>, vector<pair<double, int>>, Compare> pq;
    omp_lock_t lock;
    
public:
    ThreadSafePriorityQueue() {
        omp_init_lock(&lock);
    }
    
    ~ThreadSafePriorityQueue() {
        omp_destroy_lock(&lock);
    }
    
    void push(const pair<double, int>& item) {
        omp_set_lock(&lock);
        pq.push(item);
        omp_unset_lock(&lock);
    }
    
    bool pop(pair<double, int>& item) {
        omp_set_lock(&lock);
        if (pq.empty()) {
            omp_unset_lock(&lock);
            return false;
        }
        item = pq.top();
        pq.pop();
        omp_unset_lock(&lock);
        return true;
    }
    
    bool empty() {
        omp_set_lock(&lock);
        bool result = pq.empty();
        omp_unset_lock(&lock);
        return result;
    }
    
    size_t size() {
        omp_set_lock(&lock);
        size_t result = pq.size();
        omp_unset_lock(&lock);
        return result;
    }
};

// Synchronize ghost vertex distances using MPI collective operations
void syncGhostVertices(Graph& g, int rank, int size) {
    vector<double> send_dists(g.n + 1, INFINITY_VAL);
    
    // Copy local vertex distances to send buffer
    #pragma omp parallel for
    for (size_t i = 0; i < g.local_vertices.size(); i++) {
        int v = g.local_vertices[i];
        send_dists[v] = g.dist[v];
    }

    vector<double> recv_dists((g.n + 1) * size);
    MPI_Allgather(send_dists.data(), g.n + 1, MPI_DOUBLE, recv_dists.data(), g.n + 1, MPI_DOUBLE, MPI_COMM_WORLD);

    // Update ghost vertex distances in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < g.ghost_vertices.size(); i++) {
        int v = g.ghost_vertices[i];
        double min_dist = INFINITY_VAL;
        
        for (int j = 0; j < size; j++) {
            min_dist = min(min_dist, recv_dists[j * (g.n + 1) + v]);
        }
        
        g.dist[v] = min_dist;
    }
}

// Remove an edge from the local CSR representation (both directions for undirected)
void removeEdgeCSR(Graph& g, int u, int v) {
    // Remove v from u's adjacency if u is local
    if (u >= 1 && u <= g.n && g.partition[u] == g.partition[g.local_vertices[0]]) {
        #pragma omp parallel for
        for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; ++i) {
            if (i >= 0 && i < g.adj.size() && g.adj[i] == v) {
                g.adj[i] = -1; // Mark as removed
                g.weights[i] = INFINITY_VAL;
            }
        }
    }
    
    // Remove u from v's adjacency if v is local
    if (v >= 1 && v <= g.n && g.partition[v] == g.partition[g.local_vertices[0]]) {
        #pragma omp parallel for
        for (int i = g.row_ptr[v]; i < g.row_ptr[v + 1]; ++i) {
            if (i >= 0 && i < g.adj.size() && g.adj[i] == u) {
                g.adj[i] = -1;
                g.weights[i] = INFINITY_VAL;
            }
        }
    }
}

// Update SSSP for multiple edge changes with hybrid parallelism
void updateSSSPBatch(Graph& g, const vector<Edge>& changes, int rank, int size) {
    auto start = high_resolution_clock::now();
    
    // Build children list for deletion handling
    g.buildChildren();
    
    auto end = high_resolution_clock::now();
    if (rank == 0) {
        cout << "Build children took " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;
    }
    
    ThreadSafePriorityQueue pq;
    vector<int> affected_vertices;
    
    // Process deletions in parallel
    start = high_resolution_clock::now();
    
    // First identify all deletions
    vector<Edge> deletions;
    for (const auto& change : changes) {
        if (!change.isInsertion && (g.partition[change.u] == rank || g.partition[change.v] == rank)) {
            deletions.push_back(change);
        }
    }
    
    // Process deletions in parallel
    #pragma omp parallel
    {
        vector<int> thread_affected;
        
        #pragma omp for nowait
        for (size_t i = 0; i < deletions.size(); ++i) {
            int u = deletions[i].u;
            int v = deletions[i].v;
            
            // Remove the edge from the local CSR
            removeEdgeCSR(g, u, v);
            
            // If the deleted edge was a parent-child in the SSSP tree, mark descendants
            int x = (g.dist[u] > g.dist[v]) ? u : v;
            int y = (x == u) ? v : u;
            
            if (g.parent[x] == y && g.partition[x] == rank) {
                markDescendants(g, x, thread_affected, rank);
            }
        }
        
        // Merge thread-local affected vertices
        #pragma omp critical
        {
            affected_vertices.insert(affected_vertices.end(), thread_affected.begin(), thread_affected.end());
        }
    }
    
    end = high_resolution_clock::now();
    if (rank == 0) {
        cout << "Process deletions took " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;
    }
    
    // Initialize priority queue with affected vertices
    for (int v : affected_vertices) {
        pq.push({g.dist[v], v});
    }
    
    // Synchronize after deletions
    MPI_Barrier(MPI_COMM_WORLD);
    syncGhostVertices(g, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Process insertions in parallel
    start = high_resolution_clock::now();
    
    // First identify all insertions
    vector<Edge> insertions;
    for (const auto& change : changes) {
        if (change.isInsertion && (g.partition[change.u] == rank || g.partition[change.v] == rank)) {
            insertions.push_back(change);
        }
    }
    
    #pragma omp parallel
    {
        // Thread-local queue for new distance updates
        vector<pair<double, int>> thread_updates;
        
        #pragma omp for nowait
        for (size_t i = 0; i < insertions.size(); ++i) {
            int u = insertions[i].u;
            int v = insertions[i].v;
            double w = insertions[i].weight;
            
            // Process local vertices
            if (g.partition[v] == rank && g.dist[u] + w < g.dist[v]) {
                g.dist[v] = g.dist[u] + w;
                g.parent[v] = u;
                thread_updates.push_back({g.dist[v], v});
            }
            
            if (g.partition[u] == rank && g.dist[v] + w < g.dist[u]) {
                g.dist[u] = g.dist[v] + w;
                g.parent[u] = v;
                thread_updates.push_back({g.dist[u], u});
            }
        }
        
        // Add thread-local updates to the shared priority queue
        for (auto& update : thread_updates) {
            pq.push(update);
        }
    }
    
    end = high_resolution_clock::now();
    if (rank == 0) {
        cout << "Process insertions took " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    syncGhostVertices(g, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Update affected subgraph
    start = high_resolution_clock::now();
    bool global_change = true;
    int iteration = 0;
    
    while (global_change) {
        iteration++;
        bool local_change = false;
        
        // Process the priority queue in parallel with thread pools
        int num_threads = omp_get_max_threads();
        vector<bool> thread_changes(num_threads, false);
        
        // Continue processing until queue is empty
        while (!pq.empty()) {
            // Each thread processes CHUNK_SIZE vertices from the queue
            const int CHUNK_SIZE = 100; // Adjust based on graph properties
            vector<pair<double, int>> work_items;
            
            // Get CHUNK_SIZE items from the queue
            pair<double, int> item;
            for (int i = 0; i < CHUNK_SIZE && pq.pop(item); ++i) {
                work_items.push_back(item);
            }
            
            if (work_items.empty()) break;
            
            // Process this batch of vertices in parallel
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                vector<pair<double, int>> new_updates;
                
                #pragma omp for
                for (size_t i = 0; i < work_items.size(); ++i) {
                    double dist = work_items[i].first;
                    int z = work_items[i].second;
                    
                    // Skip if distance is outdated
                    if (dist > g.dist[z] + 1e-9) continue;
                    
                    // Process edges if the vertex is local
                    if (g.partition[z] == rank) {
                        for (int j = g.row_ptr[z]; j < g.row_ptr[z + 1]; ++j) {
                            int n = g.adj[j];
                            if (n == -1) continue; // Skip removed edges
                            
                            double w = g.weights[j];
                            double new_dist = g.dist[z] + w;
                            
                            if (new_dist < g.dist[n]) {
                                // Use atomic operation for updating distance
                                bool updated = false;
                                #pragma omp critical
                                {
                                    if (new_dist < g.dist[n]) {
                                        g.dist[n] = new_dist;
                                        g.parent[n] = z;
                                        updated = true;
                                    }
                                }
                                
                                if (updated) {
                                    new_updates.push_back({new_dist, n});
                                    if (g.partition[n] == rank) {
                                        thread_changes[thread_id] = true;
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Add new updates to priority queue
                for (auto& update : new_updates) {
                    pq.push(update);
                }
            }
        }
        
        // Check if any thread made changes
        for (bool change : thread_changes) {
            local_change |= change;
        }
        
        // Synchronize ghost vertices
        MPI_Barrier(MPI_COMM_WORLD);
        syncGhostVertices(g, rank, size);
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Check for global convergence
        int local_flag = local_change ? 1 : 0;
        int global_flag;
        MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        global_change = global_flag > 0;
    }
    
    end = high_resolution_clock::now();
    if (rank == 0) {
        cout << "Update affected subgraph took " << duration_cast<milliseconds>(end - start).count() 
             << " ms in " << iteration << " iterations" << endl;
    }
}

// Dijkstra's algorithm with OpenMP parallelism for initial SSSP
void dijkstra(Graph& g, int source) {
    try {
        if (source < 1 || source > g.n) throw runtime_error("Invalid source vertex");

        #pragma omp parallel for
        for (int i = 0; i <= g.n; i++) {
            g.dist[i] = INFINITY_VAL;
            g.parent[i] = -1;
        }
        
        g.dist[source] = 0.0;
        
        priority_queue<pair<double, int>, vector<pair<double, int>>, Compare> pq;
        vector<bool> visited(g.n + 1, false);
        
        pq.push({0.0, source});
        
        while (!pq.empty()) {
            double d = pq.top().first;
            int u = pq.top().second;
            pq.pop();
            
            if (visited[u]) continue;
            visited[u] = true;
            
            // Process neighbors in parallel
            vector<pair<double, int>> next_vertices;
            
            #pragma omp parallel
            {
                vector<pair<double, int>> thread_next;
                
                #pragma omp for nowait
                for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                    if (i < 0 || i >= g.adj.size()) continue;
                    
                    int v = g.adj[i];
                    if (v <= 0 || v > g.n || visited[v]) continue;
                    
                    double w = g.weights[i];
                    double new_dist = g.dist[u] + w;
                    
                    bool updated = false;
                    #pragma omp critical
                    {
                        if (new_dist < g.dist[v]) {
                            g.dist[v] = new_dist;
                            g.parent[v] = u;
                            updated = true;
                        }
                    }
                    
                    if (updated) {
                        thread_next.push_back({new_dist, v});
                    }
                }
                
                #pragma omp critical
                {
                    next_vertices.insert(next_vertices.end(), thread_next.begin(), thread_next.end());
                }
            }
            
            // Add vertices to the queue
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
        
        #pragma omp parallel for reduction(+:edge_count[:n+1])
        for (int i = 0; i < m; i++) {
            if (src[i] > 0 && src[i] <= n) edge_count[src[i]]++;
            if (dst[i] > 0 && dst[i] <= n) edge_count[dst[i]]++;
        }

        for (int i = 0; i < n; i++) {
            xadj[i + 1] = xadj[i] + edge_count[i + 1];
        }

        vector<int> current_pos(n + 1, 0);
        
        #pragma omp parallel
        {
            vector<pair<int, int>> thread_edges;
            
            #pragma omp for nowait
            for (int i = 0; i < m; i++) {
                int u = src[i], v = dst[i];
                if (u > 0 && u <= n && v > 0 && v <= n) {
                    thread_edges.push_back({u - 1, v - 1});
                    thread_edges.push_back({v - 1, u - 1});
                }
            }
            
            #pragma omp critical
            {
                for (auto& edge : thread_edges) {
                    int u = edge.first, v = edge.second;
                    int pos = xadj[u] + current_pos[u]++;
                    if (pos < adjncy.size()) {
                        adjncy[pos] = v;
                    }
                }
            }
        }

        vector<idx_t> part(n);
        idx_t objval;
        
        int ret = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), NULL, NULL, NULL,
                                      &nparts_idx, NULL, NULL, NULL, &objval, part.data());
                                      
        if (ret != METIS_OK) throw runtime_error("METIS partitioning failed with code " + to_string(ret));

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            partition[i + 1] = part[i];
        }

        cout << "Graph partitioned with edge-cut: " << objval << endl;
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

    changes.reserve(num_updates);
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

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    if (provided < MPI_THREAD_FUNNELED) {
        cerr << "Warning: MPI implementation does not support MPI_THREAD_FUNNELED level" << endl;
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set number of OpenMP threads
    int num_threads = 4;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);
    
    if (rank == 0) {
        cout << "Running with " << size << " MPI processes, each with " << num_threads << " OpenMP threads" << endl;
    }

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
        string input_graph = "../Dataset/sample_graph.txt";
        string updates_file = "../Dataset/updates.txt";
        
        if (argc > 2) {
            input_graph = argv[2];
        }
        
        if (argc > 3) {
            updates_file = argv[3];
        }

        // Start global timer
        double start_time = MPI_Wtime();

        // Load graph data and compute initial SSSP on rank 0
        if (rank == 0) {
            auto start_load = high_resolution_clock::now();
            
            try {
                load_data(input_graph, n, m, src, dst, weights);
                cout << "Loaded graph with " << n << " vertices and " << m << " edges" << endl;

                // Initialize full graph for Dijkstra's
                vector<int> init_partition(n + 1, 0); // Dummy partition for full graph
                auto start_dijkstra = high_resolution_clock::now();
                Graph g_full(n, m, src, dst, weights, rank, init_partition);
                dijkstra(g_full, 1);
                auto end_dijkstra = high_resolution_clock::now();
                cout << "Initial Dijkstra computation took " 
                     << duration_cast<milliseconds>(end_dijkstra - start_dijkstra).count() << " ms" << endl;
                
                initial_dist = g_full.dist;
                initial_parent = g_full.parent;

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
                
                auto end_load = high_resolution_clock::now();
                cout << "Data loading and preparation took " 
                     << duration_cast<milliseconds>(end_load - start_load).count() << " ms" << endl;
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
            src.resize(m);
            dst.resize(m);
            weights.resize(m);
        }
        
        MPI_Bcast(graph_data_buffer.data(), graph_buffer_size, MPI_BYTE, 0, MPI_COMM_WORLD);

        // Broadcast packed SSSP data size and then data
        MPI_Bcast(&sssp_buffer_size, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            sssp_data_buffer.resize(sssp_buffer_size);
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

        // Partition graph with METIS (in parallel)
        auto start_partition = high_resolution_clock::now();
        vector<int> partition = partitionGraph(n, m, src, dst, size, rank);
        auto end_partition = high_resolution_clock::now();
        
        if (rank == 0) {
            cout << "Partitioning took " 
                 << duration_cast<milliseconds>(end_partition - start_partition).count() << " ms" << endl;
        }

        // Initialize local graph with partitioned data
        auto start_construct = high_resolution_clock::now();
        Graph g(n, m, src, dst, weights, rank, partition);
        g.dist = initial_dist; // Assign broadcasted initial state
        g.parent = initial_parent; // Assign broadcasted initial state
        auto end_construct = high_resolution_clock::now();
        
        if (rank == 0) {
            cout << "Local graph construction took " 
                 << duration_cast<milliseconds>(end_construct - start_construct).count() << " ms" << endl;
        }

        // Load and broadcast edge changes
        vector<Edge> changes;
        int num_changes = 0;

        if (rank == 0) {
            try {
                load_updates(updates_file, changes);
                num_changes = changes.size();
                cout << "Loaded " << num_changes << " updates" << endl;
            } catch (const exception& e) {
                cerr << "Error loading updates: " << e.what() << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        MPI_Bcast(&num_changes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Create a custom MPI type for Edge to broadcast edge changes
        MPI_Datatype edge_type;
        int blocklengths[4] = {1, 1, 1, 1};
        MPI_Aint displacements[4];
        MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_C_BOOL};
        
        Edge temp_edge;
        MPI_Aint base_address;
        MPI_Get_address(&temp_edge, &base_address);
        MPI_Get_address(&temp_edge.u, &displacements[0]);
        MPI_Get_address(&temp_edge.v, &displacements[1]);
        MPI_Get_address(&temp_edge.weight, &displacements[2]);
        MPI_Get_address(&temp_edge.isInsertion, &displacements[3]);
        
        // Calculate relative displacements
        for (int i = 0; i < 4; i++) {
            displacements[i] = MPI_Aint_diff(displacements[i], base_address);
        }
        
        MPI_Type_create_struct(4, blocklengths, displacements, types, &edge_type);
        MPI_Type_commit(&edge_type);
        
        if (rank != 0) {
            changes.resize(num_changes);
        }
        
        // Broadcast edge changes using the custom MPI type
        MPI_Bcast(changes.data(), num_changes, edge_type, 0, MPI_COMM_WORLD);
        
        // Free the custom MPI type when done
        MPI_Type_free(&edge_type);

        int print_count = min(10, n);
        
        // Print initial state
        if (rank == 0) {
            cout << "\nBefore updates:\n";
            for (int i = 1; i <= print_count; i++) {
                cout << "Vertex " << i << ": Dist = " 
                     << (initial_dist[i] == INFINITY_VAL ? "inf" : to_string(initial_dist[i]))
                     << ", Parent = " << initial_parent[i] << "\n";
            }
            if (n > print_count) cout << "..." << endl;
        }

        // Update SSSP with timing
        MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes start together
        auto start_update = high_resolution_clock::now();
        updateSSSPBatch(g, changes, rank, size);
        MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes finish
        auto end_update = high_resolution_clock::now();
        
        if (rank == 0) {
            cout << "\nSSP update took " 
                 << duration_cast<milliseconds>(end_update - start_update).count() << " ms" << endl;
        }

        // Gather final distances and parents for output
        vector<double> global_dists(n + 1);
        vector<int> global_parents(n + 1);
        
        MPI_Reduce(g.dist.data(), global_dists.data(), n + 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(g.parent.data(), global_parents.data(), n + 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        
        // End global timer
        double end_time = MPI_Wtime();

        if (rank == 0) {
            cout << "\nTotal execution time: " << (end_time - start_time) * 1000 << " ms" << endl;
            
            cout << "\nAfter updates:\n";
            for (int i = 1; i <= print_count; i++) {
                cout << "Vertex " << i << ": Dist = " 
                     << (global_dists[i] == INFINITY_VAL ? "inf" : to_string(global_dists[i]))
                     << ", Parent = " << global_parents[i] << "\n";
            }
            if (n > print_count) cout << "..." << endl;
            
            cout << "\nTest update correctness by comparing with sequential:"
                 << "\n- Sequential: rerun same dataset with Sequential/Serial.cpp"
                 << "\n- Compare distances for vertices in 'After updates' section" << endl;
        }

    } catch (const exception& e) {
        cerr << "Fatal error on rank " << rank << ": " << e.what() << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);  
        return 1;
    }

    MPI_Finalize();
    return 0;
}
