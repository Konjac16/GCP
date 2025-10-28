#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if(argc < 3){
        cerr << "Usage: " << argv[0] << " <time_limit_seconds> <seed>\n";
        return 1;
    }
    double time_limit = atof(argv[1]);
    unsigned seed = (unsigned)strtoul(argv[2], nullptr, 10);
    (void)time_limit; srand(seed);

    int N,E,C;
    if(!(cin>>N>>E>>C)){ cerr<<"bad input\n"; return 1; }
    vector<vector<int>> adj(N);
    for(int i=0;i<E;++i){
        int u,v; cin>>u>>v;
        if(u==v) continue;
        adj[u].push_back(v); adj[v].push_back(u);
    }
    for(int i=0;i<N;++i){
        auto &a=adj[i];
        sort(a.begin(),a.end());
        a.erase(unique(a.begin(),a.end()),a.end());
    }

    vector<int> order(N); iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a,int b){ return adj[a].size()>adj[b].size(); });

    vector<int> color(N, -1);
    int usedK = 0;
    for(int v: order){
        vector<char> used(usedK, 0);
        for(int u: adj[v]) if(color[u]>=0 && color[u]<usedK) used[color[u]] = 1;
        int c=0; while(c<usedK && used[c]) ++c;
        if(c==usedK) ++usedK;
        color[v]=c;
    }

    for(int i=0;i<N;++i) cout<<color[i]<<"\n";
    return 0;
}
