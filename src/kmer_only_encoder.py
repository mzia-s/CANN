import itertools
from typing import Dict, List

class kmer_counts_encoder:
    def __init__(self, k: int):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if 4**k > 1e7:
            raise ValueError("k is too large. The number of possible k-mers exceeds 10 million.")
        self.k = k
        self.all_kmers = self.generate_kmers()
        self.counts: Dict[str, int] = {}

    def generate_kmers(self) -> List[str]:
        return [''.join(p) for p in itertools.product('ACGT', repeat=self.k)]

    def count_kmers(self, dna: str) -> Dict[str, int]:
        dna = dna.upper()
        self.counts = {kmer: 0 for kmer in self.all_kmers}
        for i in range(len(dna) - self.k + 1):
            kmer = dna[i:i + self.k]
            if kmer in self.counts:
                self.counts[kmer] += 1
        return self.counts

    def analyze(self, dna: str) -> Dict[str, int]:
        self.count_kmers(dna)
        return self.counts