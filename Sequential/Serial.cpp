#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>
#include <fstream>

using namespace std;

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
            dist.resize(n + 1, INFINITY);
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
        children.assign(n, vector<int>());
        for (int i = 0; i < n; i++) {
            if (parent[i] != -1) {
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
void markDescendants(Graph& g, int v, priority_queue<pair<double, int>, vector<pair<double, int>>, Compare>& pq) {
    g.dist[v] = INFINITY;
    g.parent[v] = -1;
    pq.push({g.dist[v], v});
    for (int child : g.children[v]) {
        markDescendants(g, child, pq);
    }
}

// Update SSSP for multiple edge changes
void updateSSSPBatch(Graph& g, const vector<Edge>& changes, const vector<bool>& isInsertion) {
    // Priority queue for vertices with their distances
    priority_queue<pair<double, int>, vector<pair<double, int>>, Compare> pq;

    // Build children list for deletion handling
    g.buildChildren();

    // Step 1: Process deletions first
    for (size_t i = 0; i < changes.size(); i++) {
        if (!isInsertion[i]) {
            int u = changes[i].u, v = changes[i].v;
            int x = (g.dist[u] > g.dist[v]) ? u : v;
            int y = (x == u) ? v : u;
            if (g.parent[x] == y) { // Edge is in SSSP tree
                markDescendants(g, x, pq);
            }
        }
    }

    // Step 2: Process insertions
    for (size_t i = 0; i < changes.size(); i++) {
        if (isInsertion[i]) {
            int u = changes[i].u, v = changes[i].v;
            double w = changes[i].weight;
            if (g.dist[u] + w < g.dist[v]) {
                g.dist[v] = g.dist[u] + w;
                g.parent[v] = u;
                pq.push({g.dist[v], v});
            }
            if (g.dist[v] + w < g.dist[u]) {
                g.dist[u] = g.dist[v] + w;
                g.parent[u] = v;
                pq.push({g.dist[u], u});
            }
        }
    }

    // Step 3: Update affected subgraph
    while (!pq.empty()) {
        auto top = pq.top();
        double dist = top.first;
        int z = top.second;
        pq.pop();

        // Skip if distance has been updated to a smaller value
        if (dist > g.dist[z] + 1e-9) continue; // Small epsilon for floating-point comparison

        // Relax edges from z
        for (int i = g.row_ptr[z]; i < g.row_ptr[z + 1]; i++) {
            int n = g.adj[i];
            double w = g.weights[i];
            if (g.dist[n] > g.dist[z] + w) {
                g.dist[n] = g.dist[z] + w;
                g.parent[n] = z;
                pq.push({g.dist[n], n});
                // Re-enqueue neighbors to ensure all paths are considered
                for (int j = g.row_ptr[n]; j < g.row_ptr[n + 1]; j++) {
                    int neighbor = g.adj[j];
                    pq.push({g.dist[neighbor], neighbor});
                }
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
    if (!file.is_open()) {
        throw runtime_error("Error opening file: " + filename);
    }

    file >> n >> m;
    src.resize(m);
    dst.resize(m);
    weights.resize(m);

    for (int i = 0; i < m; i++) {
        file >> src[i] >> dst[i] >> weights[i];
        // No need to subtract 1 from vertices as we're using 1-based indexing
    }
    file.close();
}

int main() {
    try {
        int n, m;
        vector<int> src, dst;
        vector<double> weights;

        string filename = "data.txt";
        load_data(filename, n, m, src, dst, weights);
        cout << "Loaded graph with " << n << " vertices and " << m << " edges" << endl;

        Graph g(n, m, src, dst, weights);

        // Initialize SSSP tree using Dijkstra's algorithm from vertex 1
        dijkstra(g, 1);
        cout << "Completed initial SSSP computation" << endl;

        // Example batch of updates (using 1-based vertex numbers)
        vector<Edge> changes = {
            {2, 4, 1.5},  // Insertion
            {1, 2, 1.0},  // Deletion
            {1, 4, 2.0}   // Insertion
        };
        vector<bool> isInsertion = {true, false, true};

        cout << "Before updates:\n";
        for (int i = 1; i <= n; i++) {
            cout << "Vertex " << i << ": Dist = " << g.dist[i] << ", Parent = " << g.parent[i] << "\n";
        }

        updateSSSPBatch(g, changes, isInsertion);

        cout << "\nAfter updates:\n";
        for (int i = 1; i <= n; i++) {
            cout << "Vertex " << i << ": Dist = " << g.dist[i] << ", Parent = " << g.parent[i] << "\n";
        }

    } catch (const exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return 1;
    }
    return 0;
}