#include<bits/stdc++.h>
using namespace std;

typedef vector<int> VI;
typedef vector<double> VD;
typedef vector<vector<int>> VV;
typedef vector<tuple<VI, VI, double, double>> VT;


class Apriori {
private:
	VV TDB, C, L;
	vector<map<VI, double>> Freq;
	double min_sup;
	int step;
	VT AR;

public:
	Apriori(VV _TDB, double _min_sup) {
		TDB = _TDB;
		min_sup = _min_sup;
		step = 0;

		// make 1-candidate
		set <int> C1;
		for (auto& T : TDB) {
			for (auto& item : T)
				C1.insert(item);
		}

		for (auto& item1 : C1)
			C.push_back({ item1 });

		Freq.push_back({});
	}

	VT process() {
		while (1) {
			step++;

			if (step > 1)
				C = make_Candidate();

			if (C.size() == 0)
				break;

			VD C_sup = get_Supports();
			L.clear();

			map<VI, double> now_Freq;
			for (int i = 0; i < C_sup.size(); i++) {
				if (C_sup[i] < min_sup)
					continue;

				L.push_back(C[i]);
				now_Freq[C[i]] = C_sup[i];
			}
			if (L.size() == 0)
				break;

			Freq.push_back(now_Freq);
		}

		for (auto& now_items : Freq) {
			for (auto& item : now_items) {
				make_Association(item.first, item.second, {}, {}, 0);
			}
		}

		return AR;
	}

	VV make_Candidate() {
		set<VI> check;
		VV candidate;

		// Self-Join
		for (int i = 0; i < L.size(); i++) {
			for (int j = i + 1; j < L.size(); j++) {
				auto item = merge(L[i], L[j]);
				if (item.size() != step)
					continue;
				if (check.find(item) == check.end()) {
					check.insert(item);
					// Pruning
					int k;
					for (k = 0; k < item.size(); k++) {
						auto tmp = item;
						tmp.erase(tmp.begin() + k);
						if (Freq[step - 1].find(tmp) == Freq[step - 1].end())
							break;
					}

					if (k == item.size())
						candidate.push_back(item);
				}
			}
		}

		return candidate;
	}

	void make_Association(VI item, double AB_sup, VI A, VI B, int idx) {
		if (idx == item.size()) {
			if (A.size() == 0 || B.size() == 0)
				return;
			double A_sup = Freq[A.size()][A];

			AR.push_back(make_tuple(A, B, AB_sup, AB_sup / A_sup * 100));
			return;
		}

		A.push_back(item[idx]);
		make_Association(item, AB_sup, A, B, idx + 1);
		A.pop_back();
		B.push_back(item[idx]);
		make_Association(item, AB_sup, A, B, idx + 1);
	}

	VD get_Supports() {
		VD supports;

		for (auto& item : C) {
			int support = 0;
			for (auto& T : TDB) {
				if (T.size() < item.size()) continue;

				int j = 0;
				for (int i = 0; i < T.size() && j < item.size(); i++) {
					if (T[i] == item[j])
						j++;
				}

				if (j == item.size())
					support++;
			}
			supports.push_back((double)support / TDB.size() * 100);
		}

		return supports;
	}

	VI merge(VI A, VI B) {
		VI C;

		for (int i = 0; i < A.size(); i++) {
			if (i < A.size() - 1) {
				if (A[i] == B[i])
					C.push_back(A[i]);
				else
					// If we merge these, it will be overlaped
					return {};
			}
			else {
				C.push_back((A[i] < B[i]) ? A[i] : B[i]);
				C.push_back((A[i] < B[i]) ? B[i] : A[i]);
			}
		}

		return C;
	}


};

class Input {
private:
	FILE * fo;
	VV TDB;

public:
	Input(string input_file) {
		fo = fopen(input_file.c_str(), "r");
		if (fo == NULL) {
			printf("E: Cannot open input file!\n");
			exit(0);
		}
	}

	VV make_TDB() {
		VI T;
		int num;
		char space;
		bool flag = true;

		while (fscanf(fo, "%d%c", &num, &space) != EOF) {
			T.push_back(num);
			if (space == '\n') {
				sort(T.begin(), T.end());
				TDB.push_back(T);
				T.clear();
				flag = true;
			}
			else
				flag = false;
		}
		if (!flag) {
			sort(T.begin(), T.end());
			TDB.push_back(T);
		}

		return TDB;
	}
};

class Output {
private:
	FILE * fp;

public:
	Output(string output_file) {
		fp = fopen(output_file.c_str(), "w");
		if (fp == NULL) {
			printf("E: Cannot make output file!\n");
			exit(0);
		}
	}

	void print(VT ARs) {
		for (auto AR : ARs) {
			auto A = get<0>(AR), B = get<1>(AR);
			double sup = get<2>(AR), conf = get<3>(AR);

			print_vector(A);
			print_vector(B);
			fprintf(fp, "%.2lf\t%.2lf\n", sup, conf);
		}
	}

	void print_vector(VI A) {
		fprintf(fp, "{");
		for (int i = 0; i < A.size(); i++) {
			if (i == A.size() - 1)
				fprintf(fp, "%d", A[i]);
			else
				fprintf(fp, "%d,", A[i]);
		}
		fprintf(fp, "}\t");
	}
};


int main(int argc, char* argv[]) {
	clock_t start = clock(), end;

	if (argc != 4) {
		printf("E: You should input 3 parameters! => min_sup(%%), input_file_name, output_file_name\n");
		return 0;
	}

	double min_sup = stod(string(argv[1]));
	string input_file(argv[2]), output_file(argv[3]);

	Input input(input_file);
	Apriori apriori(input.make_TDB(), min_sup);
	Output output(output_file);

	output.print(apriori.process());

	end = clock();
	printf("Total Running Time : %.3lfsec\n", (double)(end - start) / CLOCKS_PER_SEC);

	return 0;
}