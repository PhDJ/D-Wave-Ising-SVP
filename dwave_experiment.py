import numpy as np
import isingify
import pandas as pd
import minorminer as mm
import dimod
from dwave.system.composites import FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
import os


class Ising_SVP_Experiment():

    def __init__(self, dimension, k, times, n_reads, qudit_mapping):
        self.dim = dimension
        self.k = k
        self.times = times
        self.n_reads = n_reads
        self.qudit_mapping = qudit_mapping
        if self.qudit_mapping == 'bin':
            self.qubits_per_qudit = np.ceil(np.log2(2 * self.k))
        else:  # 'ham'
            self.qubits_per_qudit = 2 * self.k
        self.nqubits = self.dim * self.qubits_per_qudit
        self.t = len(self.times.values())
        """
        self.scaling_coefficient should be chosen so that big enough that you achieve almost no chain breaks,
        but at the same time small enough that you are still expanding the eigenspectrum versus the binary mapping
        # i.e. it should be less than log2(2k) times the scaling coefficient used for log, which here is 
        """
        self.scaling_coefficient = 8

    def bitstr_to_coeff_vector(self, bitstr):
        """
        Parameters
        ----------
        bitstr - a list of int Pauli-Z eigenstates (+/- 1) which can be interpretted as a bitstring through
         application of a simple operator 0.5(1-Z)

        Returns
        -------
        str - comma-delimited coefficient vector as a string
        """

        if self.qudit_mapping == "bin":
            ind = 0
            vect = []
            for i in range(self.dim):
                num = -2 ** (self.qubits_per_qudit - 1)
                for j in range(int(self.qubits_per_qudit)):
                    num += bitstr[ind] * (2 ** (j))
                    ind += 1
                vect.append(num)
            return (','.join(list(map(str, [int(y) for y in vect]))))

        else:  # 'ham'
            ind = 0
            vect = []
            for i in range(self.dim):
                num = 0
                for j in range(int(self.qubits_per_qudit)):
                    num += bitstr[ind] / 2
                    ind += 1
                vect.append(num)
            return (','.join(list(map(str, [int(y) for y in vect]))))

    def preprocess(self, lattice):
        """
        Used to map the input lattice to properties used by the D-Wave API, i.e. the interaction strengths
        Jmat, hvec, ic are explained in detail in isingify.py

        Parameters
        ----------
        lattice - numpy 2D array of type int representing a row-basis to input to the algorithm

        Returns
        -------
        """

        self.Gram = lattice@lattice.T

        if self.qudit_mapping == "bin":
            self.Jmat, self.hvec, self.ic = isingify.svp_isingcoeffs_bin(self.Gram, self.k)
        else:
            self.Jmat, self.hvec, self.ic = isingify.svp_isingcoeffs_ham(self.Gram, self.k)

        self.scale = np.max((np.abs(self.Jmat).max(), np.abs(self.hvec / 2).max())) * self.scaling_coefficient
        self.l = {i: self.hvec[i] / self.scale for i in range(self.hvec.shape[0])}
        self.q = {(i, j): self.Jmat[i, j] / self.scale for i in range(self.Jmat.shape[0]) for j in
                  range(self.Jmat.shape[1])
                  if
                  not i == j}

    def embed_graph(self):
        """
        Separate function as graph embedding can be reused so long as self.dim, self.k remain constant
        Embed graph maps the fully connected Ising model we have defined to the D-Wave Chimera topology
        """
        source_edgelist = [(i, j) for i in range(int(self.nqubits)) for j in range(i, int(self.nqubits))]
        dws = DWaveSampler(qpu=True)  # instantiate DWaveSampler
        dws_structure = dimod.child_structure_dfs(dws)
        target_edgelist = dws_structure.edgelist
        self.embedding = mm.find_embedding(source_edgelist, target_edgelist)
        physical_qubits = []
        for qubit_list in self.embedding.values():  # this loop just puts all the physical qubit indices into one list
            physical_qubits += qubit_list
        print("Number of physical qubits: ", len(physical_qubits))

    def execute_dwave(self):
        """
        Executes request to D-Wave API using predefined Ising_SVP_Experiment attributes (from preprocess())
        Saves results to response_dict attribute
        """
        self.response_dict = {}
        for i in range(self.t):
            response = FixedEmbeddingComposite(DWaveSampler(qpu=True),
                                               embedding=self.embedding).sample_ising(self.l,
                                                                                      self.q,
                                                                                      num_reads=self.n_reads,
                                                                                      annealing_time=
                                                                                      self.times[i],
                                                                                      auto_scale=False,
                                                                                      chain_strength=2.0)
            self.response_dict[self.times[i]] = response

    def postprocess(self, lattice_index):
        """
        Parameters
        ----------
        lattice_index - int index of the lattice experiment

        Returns
        -------
        N/A - processes D-Wave results and saves to CSV in file 'data'
        """
        for i in range(self.t):
            raw = list(self.response_dict[self.times[i]].data())
            if self.qudit_mapping == 'bin':
                bitlists_int = [[int(y) for y in [(1 - x) / 2 for x in list(samp[0].values())]] for samp in raw]
                bitlists_str = [''.join(list(map(str, [int(y) for y in [(1 - x) / 2 for x in list(samp[0].values())]])))
                                for
                                samp in raw]
            else:  # 'ham'
                bitlists_int = [[int(y) for y in list(samp[0].values())] for samp in raw]
                bitlists_str = [''.join(list(map(str, [int(y) for y in list(samp[0].values())])))
                                for samp in raw]
            vects = [self.bitstr_to_coeff_vector(bitlist) for bitlist in bitlists_int]
            eigenenergies = [samp[1] for samp in raw]
            occurrences = [samp[2] for samp in raw]
            chain_break = [samp[3] for samp in raw]
            df = pd.DataFrame(data=list(zip(eigenenergies, occurrences, bitlists_str, vects, chain_break)),
                              columns=['eigenenergy', 'occurrences', 'bitstring', 'coefficient_vectors', 'chain_break'])
            df['vector_lengths_sq'] = df['eigenenergy'].apply(lambda x: int(round((x * self.scale) + self.ic)))
            path = os.path.join('data',
                                '{}D_sample{}_T{}_k{}.csv'.format(self.dim, lattice_index, self.times[i], self.k))
            df.to_csv(path)

    def run(self):
        """ Preprocess, Execute, Postprocess D-Wave experiment """

        # use same embedding for all lattices of same dimension
        self.embed_graph()

        # run experiment and save results for each lattice, for a range of times
        # each lattice_index corresponds to a different random lattice to experiment on
        for lattice_index in range(2):
            lattice = np.random.randint(low=0, high=5, size=(self.dim, self.dim), dtype=int)
            self.preprocess(lattice_index, lattice)
            self.execute_dwave()
            self.postprocess(lattice_index, lattice)


def main():

    """
    Set experiment parameters

    dimension - int: lattice dimension
    k - int:  qubits per qudit
    reps - int: number of D-Wave samples collected
    qudit_mapping - str: binary or Hamming encoded qudits
    times - dict of key/value type int: keys are sweeptime indices, values are sweeptimes in microseconds
    """

    dimension = 3
    k = 4
    reps = 90
    qudit_mapping = "ham"
    times = {0: 1}

    """ Initialise Ising_SVP_Experiment object with parameters set above """
    experiment = Ising_SVP_Experiment(dimension, k, times, reps, qudit_mapping)
    experiment.run()


if __name__ == "__main__":
    main()
