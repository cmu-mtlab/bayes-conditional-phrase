#include <iostream>
#include "boost/functional/hash.hpp"
#include "corpus/corpus.h"
#include "cpyp/crp.h"
#include "cpyp/m.h"

using namespace std;
using namespace cpyp;

struct ParallelCorpus {
  Dict dict;
  set<unsigned> vocab_f, vocab_e;
  vector<vector<unsigned>> corpus_f, corpus_e;

  ParallelCorpus() {}

  explicit ParallelCorpus(string filename) {
    load(filename);
  }

  void load(string filename) {
    ReadParallelCorpusFromFile(filename.c_str(), &dict, &corpus_f, &corpus_e, &vocab_f, &vocab_e);
  }

  string line(unsigned s) {
    string r = "";
    for (unsigned w : corpus_f[s])
      r += dict.Convert(w) + " ";
    r += "|||";
    for (unsigned w : corpus_e[s])
      r += " " + dict.Convert(w);
    return r;
  }

  unsigned size() {
    assert (corpus_f.size() == corpus_e.size());
    return corpus_f.size();
  }
};

ParallelCorpus corpus; // global parallel corpus

struct TTable {
  explicit TTable(const string& fname) {
    load(fname);
  }
  double log_prob(unsigned f, unsigned e) const {
    auto it = logprobs.find(make_pair(f,e));
    if (it == logprobs.end()) {
      //cerr << "Didn't find probability p(" << corpus.dict.Convert(e) << " | " << corpus.dict.Convert(f) << ")\n";
      //abort();
      return -numeric_limits<double>::infinity();
    }
    //cerr << "Found probability to be p(" << corpus.dict.Convert(e) << " | " << corpus.dict.Convert(f) << ") = " << it->second << "\n";
    return it->second;
  }
 private:
  void load(const string& fname) {
    cerr << "Reading ttable from " << fname << endl;
    ifstream in(fname);
    string line;
    assert(in);
    string src, tgt;
    double logprob;
    while(getline(in, line)) {
      istringstream is(line);
      is >> src >> tgt >> logprob;
      logprobs[make_pair(corpus.dict.Convert(src), corpus.dict.Convert(tgt))] = logprob;
    }
  }
  unordered_map<pair<unsigned,unsigned>,double,boost::hash<pair<unsigned,unsigned>>> logprobs;
};

// Model 1.5
struct Model15 {
  double p0;
  double tension;

  Model15() : p0(0.08), tension(8.0) {}

  Model15(double p0, double tension) {
    this->p0 = p0;
    this->tension = tension;
  }

  double h(unsigned i, unsigned j, unsigned m, unsigned n) const {
    return -abs(1.0 * i / m - 1.0 * j/n);
  }

  unsigned j_up(unsigned i, unsigned m, unsigned n) const {
    return unsigned(floor(1.0 * i * n / m) + 0.5);
  }

  unsigned j_down(unsigned i, unsigned m, unsigned n) const {
    return j_up(i, m, n) + 1;
  }

  double s(double g, double r, unsigned l) const {
    return g * (1 - pow(r, l)) / (1 - r);
  }

  double Z(unsigned i, unsigned m, unsigned n) const {
    unsigned jup = j_up(i, m, n);
    unsigned jdown = j_down(i, m, n);
    double r = exp(-tension / n);
    double up_start = exp(tension * h(i, jup, m, n));
    double down_start = exp(tension * h(i, jdown, m, n));
    return s(up_start, r, jup) + s(down_start, r, n - jdown + 1);
  }

  // p(j|i, m, n)
  // j is source index, starting at 1 (0 = null)
  // i is target index, starting at 1
  // n is source length
  // m is target length
  double prob(unsigned j, unsigned i, unsigned m, unsigned n) const {
    if (j == 0) {
      assert(false && "Alignment model contains a NULL link");
      return p0;
    }
    return (1 - p0) * exp(tension * h(i, j, m, n)) / Z(i, m, n);
  }
};

struct PPSlice {
  unsigned line;
  unsigned int e_start : 8;
  unsigned int e_end : 8;
  unsigned int f_start : 8;
  unsigned int f_end : 8;
  // Covers words [start, end) on both sides

  PPSlice() {}
  PPSlice(unsigned l, unsigned fs, unsigned fe, unsigned es, unsigned ee) : line(l), e_start(es), e_end(ee), f_start(fs), f_end(fe) {}
  inline unsigned src_word(unsigned i) const {
    assert(f_end >= f_start);
    assert(i < unsigned(f_end - f_start));
    return corpus.corpus_f[line][f_start + i];
  }
  inline unsigned tgt_word(unsigned i) const {
    assert(e_end >= e_start);
    assert(i < unsigned(e_end - e_start));
    return corpus.corpus_e[line][e_start + i];
  }
  unsigned e_size() const {
    assert(e_end >= e_start);
    return e_end - e_start;
  }
  unsigned f_size() const {
    assert(f_end >= f_start);
    return f_end - f_start;
  }

  bool source_is_null() {
    return f_start == f_end;
  }

  bool target_is_null() {
    return e_start == e_end;
  }

  pair<PPSlice, PPSlice> split(unsigned f_split, unsigned e_split, bool monotonic) const { 
    // Ensure that the split point is in the middle of the phrase
    assert (e_start < e_split);
    assert (e_split < e_end);
    assert (f_start < f_split);
    assert (f_split < f_end);
 
    PPSlice first(line, monotonic ? f_start : f_split, monotonic ? f_split : f_end, e_start, e_split);
    PPSlice second(line, monotonic ? f_split : f_start, monotonic ? f_end : f_split, e_split, e_end);

    return make_pair(first, second);
  }

  PPSlice merge(const PPSlice& other) const {
    // Ensure that the two phrases are adjacent
    assert(f_start == other.f_end || f_end == other.f_start);
    assert(e_start == other.e_end || e_end == other.e_start);
    return PPSlice(line, min(f_start, other.f_start), max(f_end, other.f_end), min(e_start, other.e_start), max(e_end, other.e_end));
  }
};

std::ostream& operator<<(std::ostream& os, const PPSlice& p) {
  os << '[';
  unsigned len = p.f_size();
  for (unsigned i = 0; i < len; ++i)
    os << corpus.dict.Convert(corpus.corpus_f[p.line][p.f_start + i]) << ' ';
  if (len == 0)
    os << "NULL ";

  os << "|||";
  len = p.e_size();
  for (unsigned i = 0; i < len; ++i)
    os << ' ' << corpus.dict.Convert(corpus.corpus_e[p.line][p.e_start + i]);
  if (len == 0)
    os << " NULL";
  return os << ']';
}

struct EEquals {
  inline bool operator()(const PPSlice& a, const PPSlice& b) const {
    const unsigned elen = a.e_size();
    if (elen != b.e_size()) return false;
    const unsigned* p1 = &corpus.corpus_e[a.line][a.e_start];
    const unsigned* p2 = &corpus.corpus_e[b.line][b.e_start];
    for (unsigned i = 0; i < elen; ++i) {
      if (*p1 != *p2) return false;
      ++p1;
      ++p2;
    }
    return true;
  }
};

struct FEquals {
  inline bool operator()(const PPSlice& a, const PPSlice& b) const {
    const unsigned flen = a.f_size();
    if (flen != b.f_size()) return false;
    const unsigned* p1 = &corpus.corpus_f[a.line][a.f_start];
    const unsigned* p2 = &corpus.corpus_f[b.line][b.f_start];
    for (unsigned i = 0; i < flen; ++i) {
      if (*p1 != *p2) return false;
      ++p1;
      ++p2;
    }
    return true;
  }
};

inline bool operator==(const PPSlice& a, const PPSlice& b) {
  return EEquals()(a,b) && FEquals()(a,b);
}

bool operator!=(const PPSlice& a, const PPSlice& b) { return !(a == b); }

struct FHasher {
  size_t operator()(const PPSlice &c) const {
    return boost::hash_range(corpus.corpus_f[c.line].begin() + c.f_start, corpus.corpus_f[c.line].begin() + c.f_end);
  }
};

struct EHasher {
  size_t operator()(const PPSlice &c) const {
    return boost::hash_range(corpus.corpus_e[c.line].begin() + c.e_start, corpus.corpus_e[c.line].begin() + c.e_end);
  }
};

// the default hash function hashes on both E & F
namespace std {
  template<>
    class hash<PPSlice> {
      EHasher eh;
      FHasher fh;
      public:
        size_t operator()(const PPSlice &c) const {
          size_t h = eh(c);
          boost::hash_combine(h, fh(c));
          return h;
        }
    };
}

struct Alignment {
  vector<PPSlice> phrases; // indexed by the target position

  bool are_adjacent(unsigned i, unsigned j) const {
    const PPSlice& p1 = phrases[i];
    const PPSlice& p2 = phrases[j];
    return (p1.f_end == p2.f_start || p1.f_start == p2.f_end) &&
           (p1.e_end == p2.e_start || p1.e_start == p2.e_end);
  }

  bool is_sane() const {
    if (phrases.size() == 0)
      return true;

    int line = phrases[0].line;
    vector<bool> coverage(corpus.corpus_f[line].size(), false);

    unsigned prev_end = 0;
    for (PPSlice phrase : phrases) {
      // If the phrases are not continuous on the target side
      if (phrase.e_start != prev_end) {
        return false;
      }
      prev_end = phrase.e_end;

      // If any source word is covered more than once
      for (unsigned f = phrase.f_start; f < phrase.f_end; ++f) {
        if (coverage[f])
          return false;
        coverage[f] = true;
      }
    }

    // If the target side doesn cover the whole target sentence
    if (prev_end != corpus.corpus_e[line].size())
      return false;

    // If any source word is not covered
    if (find(coverage.begin(), coverage.end(), false) != coverage.end())
      return false;

    return true;
  }

  string toString() const {
    string r = "";
    unsigned e_len = phrases.back().e_end;
    unsigned f_len = 0;
    for (PPSlice phrase : phrases) {
      if (phrase.f_end > f_len) {
        f_len = phrase.f_end;
      }
    }

    // header line
    r += "  ";
    for (unsigned int e = 0; e < e_len; ++e) {
      r += to_string(e % 10);
    }
    r += "\n";

    // body
    for (unsigned int f = 0; f < f_len; ++f) {
      r += to_string(f % 10) + " ";
      for (PPSlice phrase : phrases) {
        char c = (f >= phrase.f_start && f < phrase.f_end) ? '*' : ' ';
        for (unsigned i = phrase.e_start; i < phrase.e_end; ++i) {
          r += c;
        }
      }
      if (f != f_len - 1)
        r += "\n";
    }

    return r;
  }

  double log_likelihood(const Model15& model) const {
    unsigned n = this->phrases.size();
    vector<unsigned> source_order(n);
    for (unsigned i = 0; i < n; ++i)
      source_order[i] = i;
    sort(source_order.begin(), source_order.end(), [this](unsigned i, unsigned j) { return this->phrases[i].f_start < this->phrases[j].f_start; });

    double llh = 0.0;
    for (unsigned i = 0; i < n; ++i) {
      unsigned j = source_order[i];
      llh += log(model.prob(j + 1, i, n, n));
    }
    return llh;
  }
};

ostream& operator<<(ostream& os, const Alignment& a) {
  os << a.toString();
  return os;
}

/*ostream& operator<<(ostream& os, const Alignment& a) {
  unsigned i = 0;
  unsigned e_len = corpus.corpus_e[a.phrases[0].line].size();
  while(i < e_len) {
    os << a.phrases[i] << endl;
    i = a.phrases[i].e_end;
  }
  return os;
}*/

struct ConditionalPhrasalModel {
  const TTable& ttable;
  const unsigned kEPS;
  double base_llh;
  unordered_map<PPSlice, crp<PPSlice>, FHasher, FEquals> cpds; // cpds[f].prob(e) gives p(e|f)

  explicit ConditionalPhrasalModel(const TTable& tt) : ttable(tt), kEPS(corpus.dict.Convert("<eps>")), base_llh() {}

  double log_p0(const PPSlice& p) const {
    const unsigned src_len = p.f_size();
    const unsigned tgt_len = p.e_size(); 
    double r = 0.0; 
    r += Md::log_poisson(tgt_len, src_len + 0.1);

    const double lp_a = log(1.0 / (src_len /*+ 1*/)); // Add +1 for NULL
    for (unsigned i=0; i < tgt_len; ++i) {
      const unsigned tgt = p.tgt_word(i);
      double p_e_i = 0;
      // The below loop should start at 0 to include NULL, or 1 to exclude it
      for (unsigned a=1; a <= src_len; ++a) {
        const unsigned src = a ? p.src_word(a - 1) : kEPS;
        const double lp_trans = ttable.log_prob(src, tgt); 
        p_e_i += exp(lp_trans + lp_a);
      }
      r += (p_e_i > 1.0e-100) ? log(p_e_i) : -numeric_limits<double>::infinity();
    }
    return r;
  }

  double log_prob(const PPSlice& p) const {
    const double base_gen = log_p0(p);
    const auto it = cpds.find(p);
    if (it == cpds.end()) return base_gen;
    return log(it->second.prob(p, exp(base_gen)));
  }

  template <typename Engine>
  void increment(const PPSlice& pp, Engine& eng) {
    auto it = cpds.find(pp);
    if (it == cpds.end()) {
      it = cpds.insert(make_pair(pp, crp<PPSlice>(1,1,1,1,0.1,0.1))).first;
    }
    const double base_gen = log_p0(pp);
    if (it->second.increment(pp, exp(base_gen), eng))
      base_llh += base_gen;
  }

  template <typename Engine>
  void decrement(const PPSlice& pp, Engine& eng) {
    auto it = cpds.find(pp);
    assert(it != cpds.end());
    if (it->second.decrement(pp, eng)) {
      base_llh -= log_p0(pp);
    }
    if (it->second.num_customers() == 0) {
      cpds.erase(it);
    }
  }

  double log_likelihood() const { 
    double llh = base_llh;
    for (auto& cpd : cpds) { 
      llh += cpd.second.log_likelihood();
    }
    return llh;
  }
};

// \lambda ~ Gamma(1,1)   : length parameter
// Generate e, given f, as follows
//   : segment f into phrases that will be translated independently
//   z = 0
//   while j < f.len:
//     z_j = 1
//     src_len ~ Poisson(\lambda)
//     j += src_len
//   z_f.len = 1
//   : z now segments f into source phrases s_{1,2,...,|s|}
//
//   : permute source phrases
//   for i = 0 .. |s|
//     a_i ~ Model1.5(i, m=|s|, n=|s|)
//
//   : generate e
//   for i = 0 .. |s|
//     e += \theta_{s_{a_i}}
struct ConditionalModel {
  const TTable ttable;
  ConditionalPhrasalModel cpm;
  double lambda; 
  vector<Alignment> z;
  Model15 model15;

  explicit ConditionalModel(const string& ttable_fname, double l) : ttable(ttable_fname), cpm(ttable), lambda(l) {}

  template <typename Engine>
  void Init(Engine& eng) {
    z.resize(corpus.corpus_e.size());
    for (unsigned i = 0; i < z.size(); ++i) {

      // We inialize to having only one phrase, but could have up to one per source word (plus NULL)
      z[i].phrases.reserve(corpus.corpus_f[i].size() + 1);
      z[i].phrases.resize(1);

      // We initialize to have a single phrase that covers the entire source and target of the sentence
      z[i].phrases[0].line = i;
      z[i].phrases[0].e_start = 0;
      z[i].phrases[0].e_end = corpus.corpus_e[i].size();
      z[i].phrases[0].f_start = 0;
      z[i].phrases[0].f_end = corpus.corpus_f[i].size();

      cpm.increment(z[i].phrases[0], eng);
      assert (z[i].is_sane());
    }
  }

  double log_likelihood() const {
    double llh = cpm.log_likelihood();
    for (const Alignment& a : z) {
      llh += a.log_likelihood(model15); 
    }
    return llh;
  }

  double phrase_log_likelihood(const PPSlice& phrase) {
    return cpm.log_prob(phrase) + Md::log_poisson(phrase.f_size(), lambda);
  }

  // Consider swapping the alignments of phrase j and phrase k within sentence #i.
  // The splitting of the sentence into phrases are frozen, along with all other alignments.
  // Note: This operator must not add or remove aligned phrase pairs!
  void swap_operator(unsigned i, unsigned j, unsigned k) {
  }

  // Consider adding or removing a phrase to sentence #i by splitting/merging at point j in the source sentence
  // and point k in the target sentence. If (j, k) was already a split point, this may also flip the monotonicity
  // of the phrases bordering (j, k), by first merging, then resplitting.
  template <typename Engine>
  void split_merge_operator(unsigned i, unsigned j, unsigned k, Engine& eng) {
    vector<unsigned> relevant_phrases;
    for (unsigned p = 0; p < z[i].phrases.size(); ++p) {
      PPSlice phrase = z[i].phrases[p];
      if ((phrase.f_start <= j && phrase.f_end >= j &&
           phrase.e_start <= k && phrase.e_end >= k)) {
          relevant_phrases.push_back(p);
      }
    }

    if (relevant_phrases.size() == 0) {
      // do nothing
    }
    else {
      // We first construct the "merged" phrase pair.
      // How we do this depends on whether (j, k) is on
      // a phrase boundary or is phrase internal
      const unsigned p = relevant_phrases[0];

      if (relevant_phrases.size() == 1) {
        // If the would-be split point is on the corner of a single phrase
        // and doesn't touch another phrase, we can't do anything, so bail.
        PPSlice& phrase = z[i].phrases[p];
        if (phrase.f_start >= j || phrase.f_end <= j ||
            phrase.e_start >= k || phrase.e_end <= k) {
          return;
        }
      }

      // Decrement the previously used phrases
      if (relevant_phrases.size() == 1) {
        cpm.decrement(z[i].phrases[p], eng);
      }
      else if (relevant_phrases.size() == 2) {
        assert(relevant_phrases[1] == p + 1);
        cpm.decrement(z[i].phrases[p], eng);
        cpm.decrement(z[i].phrases[p + 1], eng);
      }

      // The remainder of this code assumes that z[i] is in the "merged" state.
      // Thus, if it's not, we need to merge the two existing phrases.
      if (relevant_phrases.size() == 2) { 
        z[i].phrases[p] = z[i].phrases[p].merge(z[i].phrases[p + 1]);
        z[i].phrases.erase(z[i].phrases.begin() + p + 1);
      }

      // Create the merged candidate alignment
      const PPSlice& merged = z[i].phrases[p];
      const Alignment& merged_alignment = z[i];

      // Create the two split candidates
      assert (merged.f_start < j && merged.f_end > j);
      assert (merged.e_start < k && merged.e_end > k);
      pair<PPSlice, PPSlice> mono_split = merged.split(j, k, true);
      pair<PPSlice, PPSlice> rev_split = merged.split(j, k, false);

      // Create the split candidate alignments
      Alignment mono_split_alignment = merged_alignment;
      mono_split_alignment.phrases[p] = mono_split.first;
      mono_split_alignment.phrases.insert(mono_split_alignment.phrases.begin() + p + 1, mono_split.second);

      Alignment rev_split_alignment = merged_alignment;
      rev_split_alignment.phrases[p] = rev_split.first;
      rev_split_alignment.phrases.insert(rev_split_alignment.phrases.begin() + p + 1, rev_split.second);

      // Work out the three probabilities
      double merged_prob = merged_alignment.log_likelihood(model15) + phrase_log_likelihood(merged);
      double mono_split_prob = mono_split_alignment.log_likelihood(model15) + phrase_log_likelihood(mono_split.first) + phrase_log_likelihood(mono_split.second);
      double rev_split_prob = rev_split_alignment.log_likelihood(model15) + phrase_log_likelihood(rev_split.first) + phrase_log_likelihood(rev_split.second);

      // Draw one of the three probs
      vector<double> probs;
      probs.push_back(exp(merged_prob));
      probs.push_back(exp(mono_split_prob));
      probs.push_back(exp(rev_split_prob));
      multinomial_distribution<double> mult(probs);
      unsigned choice = mult(eng);
      assert (choice >= 0 && choice <= 2);

      // Update the alignment variable and increment the ttable with the phrases chosen
      if (choice == 0) {
        z[i] = merged_alignment;
        cpm.increment(merged, eng);
      }
      else if (choice == 1) {
        z[i] = mono_split_alignment;
        cpm.increment(mono_split.first, eng);
        cpm.increment(mono_split.second, eng);
      }
      else {
        z[i] = rev_split_alignment;
        cpm.increment(rev_split.first, eng);
        cpm.increment(rev_split.second, eng);
      } 
    }
  }

  // i is sentence index
  // j is an index into the source sentence
  // Considers whether to add/remove a phrase boundary at j.
  // If j is already a phrase boundary, we consider whether to move the alignment cell LEFT of j to be part of the following phrase.
  // Note: This is only possible if the phrase left of j has length of at least 2
  // If j is not already a phrase boundary, we consider whether to introduce a phrase boundary at j.
  // This cleaves the phrase pair containing j into two halves.
  // The LEFT half adjoins to the previous phrase, while the right half remains with its previous alignment.
  void flip_left_operator(unsigned i, unsigned j) {
  }

  // i is sentence index
  // j is an index into the source sentence
  // Considers whether to add/remove a phrase boundary at j.
  // If j is already a phrase boundary, we consider whether to move the alignment cell RIGHT of j to be part of the previous phrase.
  // Note: This is only possible if the phrase right of j has length of at least 2
  // If j is not already a phrase boundary, we consider whether to introduce a phrase boundary at j.
  // This cleaves the phrase pair containing j into two halves.
  // The RIGHT half adjoints to the following phrase, while the left half remains with its previous alignment.
  void flip_right_operator(unsigned i, unsigned j) {
  }

  // i is sentence index
  // j is an index into the source sentence
  // j must be on a phrase boundary
  // Let k be the target index of the phrase boundary at j
  // Consider moving the phrase boundary at (j, k) up to one space in each direction (including diagonally)
  // Note: If a phrase has length 1, its boundary may not be moved to eliminate it.
  void move_operator(unsigned i, unsigned j) {
  }
};

int main(int argc, char** argv) {
  if (argc != 4) {
    cerr << "Usage: " << argv[0] << " <file.fr-en> <ttable.txt> <nsamples>\n";
    return 1;
  }
  MT19937 eng(0);
  corpus.load(argv[1]);
  ConditionalModel cm(argv[2], 1.0);
  unsigned int samples = atoi(argv[3]);

  cm.Init(eng);
  cerr << "Model LLH:" << cm.log_likelihood() << endl;
  for (unsigned s = 0; s < corpus.size(); ++s) {
    for (unsigned i = 0; i < samples; ++i) {
      unsigned split_f = rand() % (corpus.corpus_f[s].size() - 1) + 1;
      unsigned split_e = rand() % (corpus.corpus_e[s].size() - 1) + 1;
      cm.split_merge_operator(s, split_f, split_e, eng);
    }
    cerr << corpus.line(s) << endl;
    cerr << cm.z[s] << endl << endl;
  }
  cerr << "Model LLH:" << cm.log_likelihood() << endl;
  return 0;
}
