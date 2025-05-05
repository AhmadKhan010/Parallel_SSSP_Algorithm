import argparse
import random

def read_graph(filename):
    with open(filename, 'r') as f:
        n, m = map(int, f.readline().split())
        edges = []
        for _ in range(m):
            u, v, w = f.readline().split()
            edges.append((int(u), int(v), float(w)))
    return n, m, edges

def write_updates(filename, updates):
    with open(filename, 'w') as f:
        f.write(f"{len(updates)}\n")
        for u, v, w, op in updates:
            # Format: u v w op (op: 1 for insertion, 0 for deletion)
            f.write(f"{u} {v} {w} {op}\n")

def generate_insertions(n, existing_edges, num_insertions):
    insertions = set()
    edge_set = set((min(u, v), max(u, v)) for u, v, _ in existing_edges)
    while len(insertions) < num_insertions:
        u = random.randint(1, n)
        v = random.randint(1, n)
        if u == v:
            continue
        edge = (min(u, v), max(u, v))
        if edge in edge_set or edge in insertions:
            continue
        w = round(random.uniform(1.0, 10.0), 3)
        insertions.add((u, v, w))
    return [(u, v, w, 1) for (u, v, w) in insertions]

def generate_deletions(existing_edges, num_deletions):
    deletions = random.sample(existing_edges, min(num_deletions, len(existing_edges)))
    return [(u, v, w, 0) for (u, v, w) in deletions]

def main():
    parser = argparse.ArgumentParser(description="Generate edge insertions and deletions for a graph dataset.")
    parser.add_argument('--input', required=True, help='Input graph file (txt)')
    parser.add_argument('--output', required=True, help='Output updates file (txt)')
    parser.add_argument('--insert_percent', type=float, required=True, help='Percentage of insertions (0-100)')
    parser.add_argument('--num_updates', type=int, default=None, help='Total number of updates (default: m)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    n, m, edges = read_graph(args.input)

    if args.num_updates is not None:
        total_updates = args.num_updates
    else:
        total_updates = m // 2 

    num_insertions = int(total_updates * args.insert_percent / 100)
    num_deletions = total_updates - num_insertions

    insertions = generate_insertions(n, edges, num_insertions)
    deletions = generate_deletions(edges, num_deletions)

    updates = insertions + deletions
    random.shuffle(updates)

    write_updates(args.output, updates)
    print(f"Generated {len(insertions)} insertions and {len(deletions)} deletions in {args.output}")

if __name__ == "__main__":
    main()
