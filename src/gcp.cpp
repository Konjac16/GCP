#include <bits/stdc++.h>

using namespace std;

#define ll long long;
static
const int INF_INT = 1e9;
static mt19937 rng;
inline int randint(int l, int r) {
    uniform_int_distribution < int > dist(l, r);
    return dist(rng);
}
inline double rand01() {
    uniform_real_distribution < double > dist(0.0, 1.0);
    return dist(rng);
}
struct Timer {
    using clk = chrono::steady_clock;
    clk::time_point st;
    double limit_sec;
    Timer(double sec): st(clk::now()), limit_sec(sec) {}
    inline double elapsed() const {
        return chrono::duration < double > (clk::now() - st).count();
    }
    inline bool timeup(double margin = 0.0) const {
        return elapsed() + margin >= limit_sec;
    }
    inline double left() const {
        return max(0.0, limit_sec - elapsed());
    }
};

struct Graph {
    int n, m;
    int cref;
    vector < vector < int >> adj;
    vector < pair < int, int >> edges;
    Graph(int n = 0): n(n), m(0), cref(0), adj(n) {}
};

struct Solution {
    vector < int > color;
    int k;
    int conflicts = 0;
};

static int compute_conflicts(const Graph & G,
    const vector < int > & col) {
    int cnt = 0;
    for (auto & e: G.edges) {
        int u = e.first, v = e.second;
        if (col[u] == col[v]) cnt++;
    }
    return cnt;
}

static long long hungarian_max(const vector < vector < int >> & W) {
    int n = (int) W.size();
    int m = (int) W[0].size();
    int N = max(n, m);
    vector < vector < long long >> a(N, vector < long long > (N, 0));
    int maxW = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) maxW = max(maxW, W[i][j]);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            if (i < n && j < m) a[i][j] = (long long) maxW - W[i][j];
            else a[i][j] = (long long) maxW;
        }
    // Standard O(N^3) Hungarian (min-cost assignment)
    vector < long long > u(N + 1), v(N + 1);
    vector < int > p(N + 1), way(N + 1);
    for (int i = 1; i <= N; i++) {
        p[0] = i;
        int j0 = 0;
        vector < long long > minv(N + 1, LLONG_MAX / 4);
        vector < char > used(N + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            long long delta = LLONG_MAX / 4;
            for (int j = 1; j <= N; j++)
                if (!used[j]) {
                    long long cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            for (int j = 0; j <= N; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    long long maxsum = 0;
    for (int j = 1; j <= N; j++) {
        int i = p[j];
        if (i >= 1 && i <= n && j >= 1 && j <= m) maxsum += W[i - 1][j - 1];
    }
    return maxsum;
}

static int solution_distance(const Solution & A,
    const Solution & B) {
    int n = (int) A.color.size();
    int ka = A.k, kb = B.k;
    int kk = max(ka, kb);
    vector < vector < int >> W(kk, vector < int > (kk, 0));
    for (int v = 0; v < n; ++v) {
        int ca = A.color[v];
        int cb = B.color[v];
        if (ca >= 0 && cb >= 0) {
            W[ca][cb] += 1;
        }
    }
    long long mc = hungarian_max(W);
    int d = n - (int) mc;
    if (d < 0) d = 0;
    return d;
}

static Solution dsatur(const Graph & G, int k) {
    int n = G.n;
    Solution S;
    S.k = k;
    S.color.assign(n, -1);

    vector < int > degree(n);
    for (int v = 0; v < n; ++v) degree[v] = (int) G.adj[v].size();

    vector < array < int, 2 >> sat(n); // sat[v][0]=#distinct neighbor colors; sat[v][1]=degree (for tie)
    vector < unordered_set < int >> neighColors(n);
    for (int v = 0; v < n; ++v) {
        sat[v] = {
            0,
            degree[v]
        };
    }

    auto pick = [ & ]() {
        int best = -1;
        array < int, 2 > key = {
            -1,
            -1
        };
        for (int v = 0; v < n; ++v)
            if (S.color[v] == -1) {
                array < int, 2 > cand = {
                    (int) neighColors[v].size(),
                    degree[v]
                };
                if (cand > key) {
                    key = cand;
                    best = v;
                }
            }
        if (best == -1) {
            return -1;
        }
        return best;
    };

    // first pick: max degree
    int first = -1;
    int bestdeg = -1;
    for (int v = 0; v < n; ++v)
        if (degree[v] > bestdeg) {
            bestdeg = degree[v];
            first = v;
        }
    // color first with 0
    S.color[first] = 0;
    for (int u: G.adj[first]) neighColors[u].insert(0);

    for (;;) {
        int v = pick();
        if (v < 0) break;
        int bestc = -1, bestCost = INF_INT;
        vector < int > cnt;
        cnt.assign(k, 0);
        for (int u: G.adj[v]) {
            int cu = S.color[u];
            if (cu >= 0) cnt[cu]++;
        }
        for (int c = 0; c < k; c++) {
            int cost = cnt[c];
            if (cost < bestCost) {
                bestCost = cost;
                bestc = c;
            }
        }
        S.color[v] = bestc;
        for (int u: G.adj[v]) {
            int cu = S.color[v];
            neighColors[u].insert(cu);
        }
    }
    S.conflicts = compute_conflicts(G, S.color);
    return S;
}

struct Tabu {
    int alpha_no_improve = 100000;
    int tenure_rand_max = 10; // r(10)
    int mu = 1; // l = mu * f + r(10)
};

static Solution tabu_search(const Graph & G, Solution S,
    const Timer & T,
        const Tabu & P) {
    const int n = G.n,
        k = S.k;
    // neighbor color counts: nk[v][c] = #neighbors of v currently in color c
    vector < vector < int >> nk(n, vector < int > (k, 0));
    for (int v = 0; v < n; ++v) {
        for (int u: G.adj[v]) {
            int cu = S.color[u];
            if (cu >= 0) nk[v][cu]++;
        }
    }
    int f = S.conflicts;
    int best_f = f;
    vector < int > best_col = S.color;

    // tabu_until[v][c]: iteration index until which move v->c is tabu
    vector < vector < int >> tabu_until(n, vector < int > (k, 0));
    int iter = 0;
    int no_improve = 0;

    // terminate also if time is nearly up (leave margin to print)
    while (!T.timeup(0.2)) {
        iter++;
        // collect conflict vertices
        static vector < int > confv;
        confv.clear();
        for (int v = 0; v < n; ++v) {
            if (nk[v][S.color[v]] > 0) confv.push_back(v);
        }
        if (confv.empty()) {
            // feasible k-coloring reached
            best_f = 0;
            best_col = S.color;
            break;
        }

        int bestMove_v = -1, bestMove_c = -1, bestDelta = INF_INT;
        int curIter = iter;

        // evaluate moves: only conflict vertices
        for (int idx = 0; idx < (int) confv.size(); ++idx) {
            int v = confv[idx];
            int cv = S.color[v];
            int base = nk[v][cv];
            for (int c = 0; c < k; c++)
                if (c != cv) {
                    int delta = nk[v][c] - base; // change in #conflict edges
                    bool isTabu = (tabu_until[v][c] > curIter);
                    // aspiration: allow tabu if it would improve best_f
                    if (isTabu && (f + delta) >= best_f) continue;
                    if (delta < bestDelta) {
                        bestDelta = delta;
                        bestMove_v = v;
                        bestMove_c = c;
                    } else if (delta == bestDelta) {
                        // minor random tie-break
                        if (randint(0, 1)) {
                            bestMove_v = v;
                            bestMove_c = c;
                        }
                    }
                }
        }

        if (bestMove_v == -1) {
            // all moves tabu and non-aspiring: relax by ignoring tabu once
            // fallback: pick any conflict vertex and random color
            int v = confv[randint(0, (int) confv.size() - 1)];
            int cv = S.color[v];
            int c = (cv + 1 + randint(0, k - 2)) % k;
            bestMove_v = v;
            bestMove_c = c;
            bestDelta = nk[v][c] - nk[v][cv];
        }

        // apply move
        int v = bestMove_v;
        int oldc = S.color[v], newc = bestMove_c;
        // tabu tenure
        int l = P.mu * max(1, f) + randint(1, P.tenure_rand_max);
        tabu_until[v][oldc] = iter + l; // forbid moving v back to oldc
        // update nk for neighbors
        for (int u: G.adj[v]) {
            nk[u][oldc]--;
            nk[u][newc]++;
        }
        S.color[v] = newc;
        f += bestDelta;

        // track best
        if (f < best_f) {
            best_f = f;
            best_col = S.color;
            no_improve = 0;
        } else {
            no_improve++;
            if (no_improve >= P.alpha_no_improve) break;
        }
    }

    S.color = best_col;
    S.conflicts = best_f;
    return S;
}

static Solution ampx_child(const Graph & G,
    const vector < Solution > & parents, int k) {
    int n = G.n;
    Solution C;
    C.k = k;
    C.color.assign(n, -1);

    int m = (int) parents.size();
    vector < int > forbid(m, 0);
    vector < char > used(n, 0);

    for (int step = 0; step < k; ++step) {
        int best_u = -1, best_c = -1, best_sz = -1;
        for (int u = 0; u < m; ++u) {
            if (forbid[u] > 0) continue;
            vector < int > remsz(k, 0);
            for (int v = 0; v < n; ++v) {
                int cu = parents[u].color[v];
                if (cu >= 0 && !used[v]) remsz[cu]++;
            }
            for (int c = 0; c < k; c++) {
                if (remsz[c] > best_sz) {
                    best_sz = remsz[c];
                    best_u = u;
                    best_c = c;
                }
            }
        }
        if (best_u == -1) {
            break;
        }
        for (int v = 0; v < n; ++v) {
            if (!used[v] && parents[best_u].color[v] == best_c) {
                C.color[v] = step;
                used[v] = 1;
            }
        }
        for (int u = 0; u < m; ++u)
            if (forbid[u] > 0) forbid[u]--;
        forbid[best_u] = m / 2;
    }
    vector < int > tmpCount(k, 0);
    for (int v = 0; v < n; ++v) {
        if (C.color[v] == -1) {
            fill(tmpCount.begin(), tmpCount.end(), 0);
            for (int u: G.adj[v]) {
                int cu = C.color[u];
                if (cu >= 0) tmpCount[cu]++;
            }
            int bestc = -1, bestCost = INF_INT;
            for (int c = 0; c < k; c++) {
                if (tmpCount[c] < bestCost) {
                    bestCost = tmpCount[c];
                    bestc = c;
                }
            }
            if (bestc < 0) bestc = randint(0, k - 1);
            C.color[v] = bestc;
        }
    }
    C.conflicts = compute_conflicts(G, C.color);
    return C;
}

struct PoolEntry {
    Solution S;
    double score = 0.0; // h = f + exp(b/D)
    int minDistToOthers = 0;
};

static void pool_recompute_scores(vector < PoolEntry > & P, double b) {
    int p = (int) P.size();
    // compute min distance for each (to others)
    for (int i = 0; i < p; i++) {
        int mind = INT_MAX;
        for (int j = 0; j < p; j++)
            if (i != j) {
                int d = solution_distance(P[i].S, P[j].S);
                mind = min(mind, d);
            }
        if (mind <= 0) mind = 1;
        P[i].minDistToOthers = mind;
        P[i].score = (double) P[i].S.conflicts + exp(b / (double) mind);
    }
}

static void pool_update(vector < PoolEntry > & P,
    const Solution & child, double b, double accept_worse_prob = 0.2) {
    P.push_back(PoolEntry {
        child,
        0.0,
        0
    });
    pool_recompute_scores(P, b);
    vector < int > idx(P.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [ & ](int a, int b) {
        return P[a].score > P[b].score;
    });
    int worst = idx[0];
    int child_idx = (int) P.size() - 1;
    if (worst != child_idx) {
        P.erase(P.begin() + worst);
    } else {
        if (rand01() < accept_worse_prob) {
            int second_worst = idx[1];
            P.erase(P.begin() + second_worst);
        } else {
            P.pop_back();
        }
    }
}

static Solution macol_fixed_k(const Graph & G, int k,
    const Timer & T,
        int pop_size = 20,
        int parents_min = 2,
        int parents_max = 6) {
    Tabu TSpar;
    TSpar.alpha_no_improve = 100000;
    TSpar.tenure_rand_max = 10;
    TSpar.mu = 1;

    vector < PoolEntry > pool;
    pool.reserve(pop_size);
    for (int i = 0; i < pop_size; i++) {
        if (T.timeup(0.2)) break;
        Solution S0 = dsatur(G, k);
        S0 = tabu_search(G, S0, T, TSpar);
        pool.push_back(PoolEntry {
            S0,
            0.0,
            0
        });
    }
    if (pool.empty()) {
        Solution S;
        S.k = k;
        S.color.assign(G.n, 0);
        S.conflicts = compute_conflicts(G, S.color);
        return S;
    }
    double b = 0.08 * (double) G.n;
    pool_recompute_scores(pool, b);

    Solution best = pool[0].S;
    for (auto & e: pool)
        if (e.S.conflicts < best.conflicts) best = e.S;

    while (!T.timeup(0.2)) {
        if (best.conflicts == 0) break;

        int m = randint(parents_min, parents_max);
        m = min(m, (int) pool.size());
        vector < int > pick;
        vector < int > ids(pool.size());
        iota(ids.begin(), ids.end(), 0);
        shuffle(ids.begin(), ids.end(), rng);
        for (int i = 0; i < m; i++) pick.push_back(ids[i]);

        vector < Solution > parents;
        parents.reserve(m);
        for (int id: pick) parents.push_back(pool[id].S);

        // crossover -> child
        Solution child = ampx_child(G, parents, k);
        // local search
        child = tabu_search(G, child, T, TSpar);

        // update best
        if (child.conflicts < best.conflicts) best = child;

        // pool update
        pool_update(pool, child, b, 0.2);
    }
    return best;
}

int main(int argc, char ** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 3) {
        cerr << "Usage: gcp.exe <time_limit_sec> <seed>\n";
        return 0;
    }
    double time_limit = atof(argv[1]);
    unsigned int seed = (unsigned int) strtoul(argv[2], nullptr, 10);
    rng.seed(seed);

    // read graph
    int N, E, Cref;
    if (!(cin >> N >> E >> Cref)) {
        cerr << "Bad input header\n";
        return 0;
    }
    Graph G(N);
    G.m = E;
    G.cref = Cref;
    G.adj.assign(N, {});
    G.edges.reserve(E);
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        if (u < 0 || u >= N || v < 0 || v >= N) {
            cerr << "Edge out of range " << u << " " << v << "\n";
            continue;
        }
        if (u == v) continue;
        G.adj[u].push_back(v);
        G.adj[v].push_back(u);
        if (u < v) G.edges.emplace_back(u, v);
        else G.edges.emplace_back(v, u);
    }

    Timer T(time_limit);

    int k = max(1, Cref);
    Solution Best;
    Best.k = k;
    Best.color.assign(N, 0);
    Best.conflicts = compute_conflicts(G, Best.color);

    for (; k >= 1 && !T.timeup(0.5);) {
        Solution cur = macol_fixed_k(G, k, T);
        if (cur.conflicts < Best.conflicts || (Best.k > cur.k && cur.conflicts == 0)) {
            Best = cur;
        }
        if (cur.conflicts == 0) {
            k--;
        }
    }
    for (int i = 0; i < N; i++) {
        int c = Best.color[i];
        if (c < 0) c = 0;
        cout << c << "\n";
    }
    return 0;
}