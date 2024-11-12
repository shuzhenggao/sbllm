

#include <bits/stdc++.h>

using namespace std;



const int maxs = 4096;

int dp[maxs];



int main() {

	memset(dp, 0x3f, 16384);

	dp[0] = 0;

	int n, m;

	cin >> n >> m;

	while (m--) {

		int a, b, s = 0;

		cin >> a >> b;

		while (b--) {

			int c;

			cin >> c;

			s ^= 1 << c - 1;

		}

		dp[s] = min(dp[s], a);

	}

	for (int i = 0; i != n; ++i) for (int all = (1 << n) - 1 ^ 1 << i, sub = all; sub; sub = sub - 1 & all)

		dp[sub] = min(dp[sub], dp[sub ^ 1 << i]);

	for (int s = 0; s != (1 << n); ++s) if (s != (s & -s))

		for (int t = s - 1 & s; t != s; t = t - 1 & s)

			dp[s] = min(dp[s], dp[t] + dp[s ^ t]);

	int mask = (1 << n) - 1;

	if (dp[mask] == 0x3f3f3f3f) dp[mask] = -1;

	cout << dp[mask] << endl;

}