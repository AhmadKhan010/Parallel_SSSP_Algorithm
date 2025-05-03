#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>

using namespace std;

const double INFINITY = numeric_limits<double>::infinity();

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
    int m;                  // Number of edges

    Graph(int n, int m, const vector<int>& src, const vector<int>& dst, const vector<double>& weights) {
        this->n = n;
        this->m = m;
        dist.resize(n, INFINITY);
        parent.resize(n, -1);
        row_ptr.resize(n + 1, 0);
        adj.resize(m);
        this->weights.resize(m);

        // Build CSR: count edges per vertex
        vector<int> edge_count(n, 0);
        for (int i = 0; i < m; i++) {
            edge_count[src[i]]++;
        }
        row_ptr[0] = 0;
        for (int i = 0; i < n; i++) {
            row_ptr[i + 1] = row_ptr[i] + edge_count[i];
        }

        // Fill adjacency and weights
        vector<int> current_pos(n, 0);
        for (int i = 0; i < m; i++) {
            int v = src[i];
            int pos = row_ptr[v] + current_pos[v]++;
            adj[pos] = dst[i];
            this->weights[pos] = weights[i];
        }
    }
};

// Custom comparator for min-heap priority queue
struct Compare {
    bool operator()(const pair<double, int>& a, const pair<double, int>& b) {
        return a.first > b.first; // Min-heap based on distance
    }
};

// Update SSSP for multiple edge changes
void updateSSSPBatch(Graph& g, const vector<Edge>& changes, const vector<bool>& isInsertion) {
    // Priority queue for vertices with their distances
    priority_queue<pair<double, int>, vector<pair<double, int>>, Compare> pq;
    vector<bool> affected(g.n, false);
    vector<bool> inQueue(g.n, false); // Track vertices in priority queue

    // Step 1: Process deletions first
    for (size_t i = 0; i < changes.size(); i++) {
        if (!isInsertion[i]) {
            int u = changes[i].u, v = changes[i].v;
            int x = (g.dist[u] > g.dist[v]) ? u : v;
            int y = (x == u) ? v : u;
            if (g.parent[x] == y) { // Edge is in SSSP tree
                g.dist[x] = INFINITY;
                g.parent[x] = -1;
                affected[x] = true;
                if (!inQueue[x]) {
                    pq.push({g.dist[x], x});
                    inQueue[x] = true;
                }
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
                affected[v] = true;
                if (!inQueue[v]) {
                    pq.push({g.dist[v], v});
                    inQueue[v] = true;
                }
            }
            if (g.dist[v] + w < g.dist[u]) {
                g.dist[u] = g.dist[v] + w;
                g.parent[u] = v;
                affected[u] = true;
                if (!inQueue[u]) {
                    pq.push({g.dist[u], u});
                    inQueue[u] = true;
                }
            }
        }
    }

    // Step 3: Update affected subgraph
    while (!pq.empty()) {
        auto [dist, z] = pq.top();
        pq.pop();
        inQueue[z] = false;

        // Skip if distance has been updated to a smaller value
        if (dist > g.dist[z]) continue;

        affected[z] = false;

        // Relax edges from z
        for (int i = g.row_ptr[z]; i < g.row_ptr[z + 1]; i++) {
            int n = g.adj[i];
            double w = g.weights[i];
            if (g.dist[n] > g.dist[z] + w) {
                g.dist[n] = g.dist[z] + w;
                g.parent[n] = z;
                if (!inQueue[n]) {
                    pq.push({g.dist[n], n});
                    inQueue[n] = true;
                    affected[n] = true;
                }
            }
        }

        // Re-enqueue neighbors if z was updated
        for (int i = g.row_ptr[z]; i < g.row_ptr[z + 1]; i++) {
            int n = g.adj[i];
            if (!affected[n] && !inQueue[n]) {
                pq.push({g.dist[n], n});
                inQueue[n] = true;
                affected[n] = true;
            }
        }
    }
}

int main() {
    // Example graph: 5 vertices, 6 edges (undirected)
    int n = 5, m = 6;
    vector<int> src = {0, 0, 1, 1, 2, 3}; // Source vertices
    vector<int> dst = {1, 2, 0, 3, 3, 4};  // Destination vertices
    vector<double> weights = {1.0, 4.0, 1.0, 2.0, 2.0, 1.0}; // Edge weights

    Graph g(n, m, src, dst, weights);

    // Set initial SSSP from source vertex 0
    g.dist[0] = 0.0;
    g.parent[0] = -1;
    g.dist[1] = 1.0;
    g.parent[1] = 0;
    g.dist[2] = 4.0;
    g.parent[2] = 0;
    g.dist[3] = 3.0;
    g.parent[3] = 1;
    g.dist[4] = 4.0;
    g.parent[4] = 3;

    // Example batch of updates
    vector<Edge> changes = {
        {2, 4, 1.5}, // Insertion: edge (2,4) with weight 1.5
        {0, 1, 1.0}, // Deletion: edge (0,1)
        {1, 4, 2.0}  // Insertion: edge (1,4) with weight 2.0
    };
    vector<bool> isInsertion = {true, false, true};

    cout << "Before updates:\n";
    for (int i = 0; i < n; i++) {
        cout << "Vertex " << i << ": Dist = " << g.dist[i] << ", Parent = " << g.parent[i] << "\n";
    }

    // Update SSSP
    updateSSSPBatch(g, changes, isInsertion);

    cout << "\nAfter updates:\n";
    for (int i = 0; i < n; i++) {
        cout << "Vertex " << i << ": Dist = " << g.dist[i] << ", Parent = " << g.parent[i] << "\n";
    }

    return 0;
}