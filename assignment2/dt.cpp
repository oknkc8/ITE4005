#include<bits/stdc++.h>
#include<unordered_map>
using namespace std;

#define MAXINPUTLENGTH 10000

typedef vector<string> VS;
typedef vector<vector<string>> VVS;
typedef long double ld;
typedef unordered_map<string, int> UMI;
typedef unordered_map<string, VVS> UMVVS;

class Table {
public:
	VS attr_name;
	VVS data;
};

class Node {
public:
	vector<Node*> child;
	VS attrs;
	string attr_name;
	int attr_idx;
	UMI lable_count;

	Node(VS attr_name) {
		for (auto attr : attr_name) {
			lable_count[attr] = 0;
		}
	}

	bool is_leaf() {
		if (child.size() == 0) {
			return true;
		}
		else {
			return false;
		}
	}
};

class Decision_Tree {
private:
	Table table;
	int col;
	Node* Tree;

	void make_Tree(Node* now, VVS t) {
		if (is_homogeneous(t)) {
			now->attr_name = t[0][col - 1];
			now->lable_count[now->attr_name]++;
			return;
		}

		// Select the Splitting Attribute
		ld max_gain_ratio = 0;
		int max_attr_idx = 0;
		for (int attr_idx = 0; attr_idx < col - 1; attr_idx++) {
			ld now_gain_ratio = gain_ratio(t, attr_idx);

			if (now_gain_ratio > max_gain_ratio) {
				max_gain_ratio = now_gain_ratio;
				max_attr_idx = attr_idx;
			}
		}

		now->attr_name = table.attr_name[max_attr_idx];
		now->attr_idx = max_attr_idx;

		UMVVS child_rows = make_child(t, max_attr_idx);
		for (auto child_row : child_rows) {
			string attr_value = child_row.first;
			Node* child = new Node(table.attr_name);
			now->attrs.push_back(attr_value);

			make_Tree(child, child_row.second);
			
			now->child.push_back(child);
			for (auto label : child->lable_count) {
				now->lable_count[label.first] += label.second;
			}
		}
	}

	bool is_homogeneous(VVS t) {
		if (get_entropy(t) == 0) {
			return true;
		}
		else {
			return false;
		}
	}

	UMVVS make_child(VVS t, int attr_idx) {
		vector<VVS> child;
		UMVVS hash_row;
		int D = t.size();
		int idx = attr_idx;

		for (int i = 0; i < D; i++) {
			hash_row[t[i][idx]].push_back(t[i]);
		}

		return hash_row;
	}

	ld get_entropy(VVS t) {
		UMI hash;
		ld entropy = 0;
		int D = t.size();
		int idx = col - 1;

		for (int i = 0; i < D; i++) {
			hash[t[i][idx]]++;
		}

		for (auto i : hash) {
			ld pi = (ld)i.second / (ld)D;
			entropy += (pi * log2l(pi));
		}

		return -entropy;
	}

	ld get_entropy_with_attr(VVS t, int attr_idx) {
		UMI hash;
		UMVVS hash_row;
		ld entropy = 0;
		int D = t.size();
		int idx = attr_idx;

		for (int i = 0; i < D; i++) {
			hash[t[i][idx]]++;
			hash_row[t[i][idx]].push_back(t[i]);
		}

		for (auto i : hash) {
			ld attr_entropy = get_entropy(hash_row[i.first]);
			ld pi = (ld)i.second / (ld)D;
			entropy += (pi * attr_entropy);
		}

		return entropy;
	}

	ld gain(VVS t, int attr_idx) {
		return get_entropy(t) - get_entropy_with_attr(t, attr_idx);
	}

	ld split_info(VVS t, int attr_idx) {
		UMI hash;
		ld entropy = 0;
		int D = t.size();
		int idx = attr_idx;

		for (int i = 0; i < D; i++) {
			hash[t[i][idx]]++;
		}

		for (auto i : hash) {
			ld pi = (ld)i.second / (ld)D;
			entropy += (pi * log2l(pi));
		}

		return -entropy;
	}

	ld gain_ratio(VVS t, int attr_idx) {
		ld g = gain(t, attr_idx);
		ld si = split_info(t, attr_idx);
		
		if (si == 0) {
			return -1;
		}
		else {
			return g / si;
		}
	}

	string make_decision(Node* now, VS row) {
		if (now->is_leaf()) {
			return now->attr_name;
		}

		for (int i = 0; i < now->attrs.size(); i++) {
			string attr_value = now->attrs[i];
			if (attr_value == row[now->attr_idx]) {
				return make_decision(now->child[i], row);
			}
		}

		// If no decision
		// get the most label in Node
		int count_max = 0;
		string decision;
		for (auto label : now->lable_count) {
			if (count_max < label.second) {
				count_max = label.second;
				decision = label.first;
			}
		}
		return decision;
	}

public:
	Decision_Tree(Table train_table) {
		table = train_table;
		col = table.attr_name.size();
	}

	void train() {
		Tree = new Node(table.attr_name);
		make_Tree(Tree, table.data);
	}

	Table test(Table test_table) {
		VVS& t = test_table.data;

		for (int i = 0; i < t.size(); i++) {
			t[i].push_back(make_decision(Tree, t[i]));
		}

		test_table.attr_name.push_back(*(table.attr_name.end() - 1));

		return test_table;
	}
};

class Input {
private:
	FILE * fo;
	Table table;

public:
	Input(string input_file) {
		fo = fopen(input_file.c_str(), "r");
		if (fo == NULL) {
			printf("E: Cannot open input file!\n");
			exit(0);
		}
		parse();
	}

	void parse() {
		char input_str[MAXINPUTLENGTH];
		bool is_first_row = true;
		while (fgets(input_str, MAXINPUTLENGTH, fo) != NULL) {
			string str(input_str);
			VS row;
			int idx = 0;

			for (int i = 0; i < str.size(); i++) {
				if (str[i] == '\t' || str[i] == '\n' || str[i] == '\r') {
					if (i - idx == 0) {
						continue;
					}
					string value = str.substr(idx, i - idx);
					row.push_back(value);
					idx = i + 1;
				}
			}

			if (is_first_row) {
				table.attr_name = row;
				is_first_row = false;
			}
			else {
				table.data.push_back(row);
			}
		}
	}

	Table get_Table() {
		return table;
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

	void print(Table result_table) {
		// print attribute_name
		for (int i = 0; i < result_table.attr_name.size(); i++) {
			if (i == result_table.attr_name.size() - 1) {
				fprintf(fp, "%s\n", result_table.attr_name[i].c_str());
			}
			else {
				fprintf(fp, "%s\t", result_table.attr_name[i].c_str());
			}
		}

		// print attribute
		for (auto row : result_table.data) {
			for (int i = 0; i < row.size(); i++) {
				if (i == row.size() - 1) {
					fprintf(fp, "%s\n", row[i].c_str());
				}
				else {
					fprintf(fp, "%s\t", row[i].c_str());
				}
			}
		}
	}
};

int main(int argc, char* argv[]) {
	clock_t start = clock(), end;

	if (argc != 4) {
		printf("E: You should input 3 parameters! => min_sup(%%), input_file_name, output_file_name\n");
		return 0;
	}

	string train_file(argv[1]), test_file(argv[2]), output_file(argv[3]);

	Input train_input(train_file), test_input(test_file);
	Output output(output_file);

	Table train_table = train_input.get_Table(), test_table = test_input.get_Table();

	Decision_Tree DT(train_table);
	DT.train();
	output.print(DT.test(test_table));

	end = clock();
	printf("Total Running Time : %.3lfsec\n", (double)(end - start) / CLOCKS_PER_SEC);

	return 0;
}