#include<bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
	srand((unsigned int)time(NULL));

	int num_items = stoi(string(argv[1])), num_transactions = stoi(string(argv[2]));

	string input_file = "input/input_" + to_string(num_items) + "_" + to_string(num_transactions) + ".txt";

	FILE *fp = fopen(input_file.c_str(), "w");

	while (num_transactions--) {
		int num = rand() % num_items + 1;
		vector<int> v(num_items, 0);
		while (num--) {
			int t = rand() % num_items;
			while(v[t])
				t = rand() % num_items;

			if (num == 0) fprintf(fp, "%d\n", t);
			else fprintf(fp, "%d\t", t);
		}
	}	

	return 0;
}