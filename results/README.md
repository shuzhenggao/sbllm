

## Cpp I

| Approach | OPT(top1) | SP(top1) | OPT(top3) | SP(top3) | OPT(top5) | SP(top5) |
|----------|-----------|----------|-----------|----------|-----------|----------|
| 1        | 6.61      | 16.89    | 8.21      | 18.45    | 8.89      | 19.61    |
| 2        | 5.70      | 17.53    | 6.27      | 18.97    | 6.96      | 20.01    |
| 3        | 5.70      | 19.45    | 6.27      | 20.33    | 6.61      | 21.21    |
| 4        | 5.93      | 20.84    | 7.75      | 21.83    | 8.21      | 22.26    |
| 5        | 5.70      | 19.70    | 6.96      | 20.83    | 7.41      | 21.99    |


## Cpp N_s

| Approach | OPT(top1) | SP(top1) | OPT(top3) | SP(top3) | OPT(top5) | SP(top5) |
|----------|-----------|----------|-----------|----------|-----------|----------|
| 1        | 5.82      | 18.45    | 6.61      | 19.67    | 7.75      | 20.73    |
| 2        | 5.82      | 18.27    | 7.18      | 19.89    | 7.75      | 21.60    |
| 3        | 5.93      | 20.84    | 7.75      | 21.83    | 8.21      | 22.26    |
| 4        | 5.93      | 16.45    | 7.07      | 18.16    | 7.87      | 18.79    |
| 5        | 5.47      | 16.19    | 6.50      | 17.63    | 6.73      | 18.12    |



## Python N_s

| Approach | OPT(top1) | SP(top1) | OPT(top3) | SP(top3) | OPT(top5) | SP(top5) |
|----------|-----------|----------|-----------|----------|-----------|----------|
| 1        | 35.29     | 71.96    | 37.02     | 76.14    | 37.73     | 77.80    |
| 2        | 35.50     | 73.45    | 37.12     | 78.13    | 37.73     | 79.68    |
| 3        | 35.60     | 75.54    | 37.32     | 78.74    | 37.83     | 79.93    |
| 4        | 34.58     | 75.26    | 37.32     | 79.83    | 38.44     | 82.08    |
| 5        | 35.19     | 75.89    | 36.61     | 78.90    | 37.93     | 81.75    |


## Python I

| Approach | OPT(top1) | SP(top1) | OPT(top3) | SP(top3) | OPT(top5) | SP(top5) |
|----------|-----------|----------|-----------|----------|-----------|----------|
| 1        | 35.50     | 73.97    | 37.53     | 77.35    | 38.44     | 79.52    |
| 2        | 36.21     | 76.39    | 38.03     | 79.68    | 38.84     | 81.50    |
| 3        | 34.58     | 75.26    | 37.32     | 79.83    | 38.44     | 82.08    |
| 4        | 35.70     | 75.12    | 37.53     | 78.66    | 38.03     | 79.87    |
| 5        | 34.79     | 72.69    | 36.92     | 76.66    | 37.22     | 77.66    |




## Cpp-top3

| Approach    | NC        | NO       | NH        | FH       | Unique number |
|-------------|-----------|----------|-----------|----------|---------------|
| Instruction | 20.64     | 76.17    | 2.51     | 0.68    | 3          |
| ICL         | 33.75     | 63.63    | 1.60     | 1.03    | 3          |
| RAG         | 33.07     | 64.42    | 2.05     | 0.46    | 3          |
| COT         | 30.78     | 62.71    | 4.56     | 1.94    | 11         |
| SBLLM       | 26.34     | 65.91    | 4.45     | 3.31    | 15         |

## Cpp-top5

| Approach    | NC        | NO       | NH        | FH       | Unique number |
|-------------|-----------|----------|-----------|----------|---------------|
| Instruction | 18.36     | 76.40    | 3.88     | 1.37    | 2         |
| ICL         | 28.51     | 67.84    | 2.51     | 1.14    | 1         |
| RAG         | 30.33     | 65.45    | 3.08     | 1.14    | 5         |
| COT         | 25.88     | 66.70    | 5.13     | 2.28    | 13         |
| SBLLM       | 25.31     | 66.48    | 4.67     | 3.53    | 18         |


## Python-top3

| Approach    | NC        | NO       | NH        | FH       | Unique number |
|-------------|-----------|----------|-----------|----------|---------------|
| Instruction | 24.04     | 67.85    | 65.92     | 1.52     | 14         |
| ICL         | 20.89     | 63.49    | 12.78     | 2.84     | 16         |
| RAG         | 30.73     | 53.55    | 13.59     | 2.13     | 9          |
| COT         | 27.99     | 36.21    | 25.63     | 9.16     | 40         |
| SBLLM       | 20.18     | 41.07    | 28.70     | 10.04    | 65         |

## Python-top5

| Approach    | NC        | NO       | NH        | FH       | Unique number |
|-------------|-----------|----------|-----------|----------|---------------|
| Instruction | 21.50     | 68.36    | 8.42      | 1.72    | 11         |
| ICL         | 17.44     | 63.59    | 14.60     | 4.36    | 15         |
| RAG         | 28.30     | 48.88    | 17.95     | 4.87    | 32         |
| COT         | 22.82     | 38.84    | 28.60     | 9.74     | 28         |
| SBLLM       | 18.56     | 40.97    | 29.61     | 10.85    | 52         |



## Complete Case of Fig. 10
a. Slow code
```cpp
main() {
  int A,B,C;
  cin >>A>>B>>C;
  int A2=A;
  vector<bool> B2(B);
  for(int i=0;i<1e7;++i){
    B2[A2%B]=true;
    A2+=A;
  }
  cout<<(B2[C]?"YES":"NO")<<endl;
}
```
b. Optimized code 1 in iteration 2
```cpp
int main() {
  int A, B, C;
  cin >> A >> B >> C;
  int A2 = A % B;
  vector<bool> B2(B);
  for (int i = 0; i < B; ++i) {
    B2[A2] = true;
    A2 = (A2 + A) % B;
  }
  cout<<(B2[C]?"YES":"NO")<<endl;
  return 0;
}
```
c. Optimized code 2 in iteration 2
```cpp
int main() {
  int A, B, C;
  cin >> A >> B >> C;
  int A2 = A % B;
  vector<bool> B2(B);
  for (int i = 0; i <1e7; ++i) {
    B2[A2%B] = true;
    if (A2%B == C) {
      cout << "YES" << endl;
      return 0;
    }
    A2+=A;
  }
  cout << "NO" << endl;
  return 0;
}
```
d. Optimized code 3 in iteration 3
```cpp
int main() {
  int A, B, C;
  cin >> A >> B >> C;
  int A2 = A % B;
  vector<bool> B2(B);
  for (int i = 0; i < B; ++i) {
    B2[A2] = true;
    if (A2 == C) {
      cout << "YES" << endl;
      return 0;
    }
    A2 = (A2 + A) % B;
  }
  cout << "NO" << endl;
  return 0;
}
```


## Complete Case of Fig. 11
a. Slow code
```cpp
int main (){
    int N,M,p[100000];
    string S[100000];
    bool ac_check[100000];
    bool is_ac[100000];
    for(int i = 0;i<M;i++){
        ac_check[i] = 0;
        is_ac[i] = 0;
    }
    cin >> N;
    cin >> M;
    for(int i = 0;i<M;i++){
        cin >> p[i] >> S[i];
      	 if(S[i] == "AC")
        is_ac[p[i]] = 1;
    }
    int ac = 0;
    int wa = 0;
    for(int i = 0;i<M;i++){
        if(S[i]== "AC"&&!ac_check[p[i]]){
            ac_check[p[i]] = 1;
            ac++;
        }
        if(S[i]== "WA"&&!ac_check[p[i]]&&is_ac[p[i]]){
            wa++;
        }
    }
    cout << ac << " " << wa << endl;
    return 0;
}
```
b. Optimized code 1
```cpp
int main() {
    int N, M;
    cin >> N >> M;
    unordered_map<int, pair<bool, bool>> problems;
    int ac = 0;
    int wa = 0;
    for (int i = 0; i < M; i++) {
        int p;
        string S;
        cin >> p >> S;
        if (S == "AC") {
            problems[p].first = true;
            ac++;
            if (problems[p].second) wa++;
        } else if (S == "WA") {
            problems[p].second = true;
        }
    }
    cout << ac << " " << wa << endl;
    return 0;
}
```
c. Optimized code 2
```cpp
int main() {
    int N, M;
    cin >> N >> M;
    unordered_map<int, pair<bool, bool>> problems;
    int ac = 0;
    int wa = 0;
    for (int i = 0; i < M; i++) {
        int p;
        string S;
        cin >> p >> S;
        if (problems[p].first) continue;
        if (S == "AC") {
            problems[p].first = true;
            ac++;
            if (problems[p].second) wa++;
        } else if (S == "WA") {
            problems[p].second = true;
        }
    }
    cout << ac << " " << wa << endl;
    return 0;
}
```












