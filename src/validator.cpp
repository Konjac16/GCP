#include <bits/stdc++.h>
using namespace std;

struct Instance {
    int N=0, E=0, Cref=0;
    vector<pair<int,int>> edges;
};

static bool read_instance(const string& path, Instance& ins){
    ifstream fin(path);
    if(!fin) return false;
    fin >> ins.N >> ins.E >> ins.Cref;
    ins.edges.resize(ins.E);
    for(int i=0;i<ins.E;i++){
        int u,v; fin>>u>>v;
        ins.edges[i]={u,v};
    }
    return true;
}

static bool read_solution(const string& path, int N, vector<long long>& col, string& err){
    ifstream fin(path);
    if(!fin){ err="cannot open solution"; return false; }
    col.clear(); col.reserve(N);
    long long x;
    while(fin>>x){
        if(x<0){ err="negative color"; return false; }
        col.push_back(x);
    }
    if((int)col.size()!=N){
        err = "line count mismatch: got " + to_string(col.size()) + ", need " + to_string(N);
        return false;
    }
    return true;
}

int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if(argc < 3){
        cerr << "Usage: validator <instance> <solution>\n";
        return 2;
    }
    string inst = argv[1], sol = argv[2];

    Instance ins;
    if(!read_instance(inst, ins)){
        cerr << "ERROR: cannot read instance: " << inst << "\n";
        return 2;
    }

    vector<long long> col;
    string err;
    if(!read_solution(sol, ins.N, col, err)){
        cout << "N="<<ins.N<<" E="<<ins.E<<" Cref="<<ins.Cref
             << " colors=-1 conflicts=-1\n";
        cerr << "INVALID solution: " << err << "\n";
        return 1;
    }

    long long conflicts = 0;
    for(auto [u,v]: ins.edges){
        if(u<0 || u>=ins.N || v<0 || v>=ins.N){ cerr<<"edge out of range\n"; return 1; }
        if(col[u] == col[v]) conflicts++;
    }
    unordered_set<long long> S; S.reserve(ins.N*2);
    for(auto c: col) S.insert(c);
    long long colors = (long long)S.size();

    cout << "N="<<ins.N<<" E="<<ins.E<<" Cref="<<ins.Cref
         << " colors="<<colors<<" conflicts="<<conflicts<<"\n";


    cout << "[REPORT] instance: " << inst << "\n";
    cout << "         solution: " << sol << "\n";
    cout << "         vertices: " << ins.N << ", edges: " << ins.E << "\n";
    cout << "         colors:   " << colors << " (Cref=" << ins.Cref << ")\n";
    cout << "         conflicts:" << conflicts << "\n";

    return 0;
}
