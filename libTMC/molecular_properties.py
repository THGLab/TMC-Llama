"""
    A collection of functions to summarize ring structures
    and to calculate related properties for TMC molecules.

    Key:
        RDKit doesn't consider dative bonds when counting
        rings in TMC molecules, so the results are effectively
        ligand structural properties. The algorithms can be
        changed through GSSSR in RDKit by setting
        "includeDativeBonds" to True. In this case, all RDKit
        calculations, including the algorithms in GetRingInfo(),
        will keep dative bonds and the setting will remain
        in-place unless GSSSR is altered again. 
"""

# RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Mol, EditableMol

def find_all_tms(mol, use_transition_metal=False):
    """
        Identify all transition metals (or metals)
        in a molecule (or complex)
    """
    outmol = Mol(mol)
    tm_idx_list = []
    for atom in outmol.GetAtoms():
        if use_transition_metal:
            if _is_transition_metal(atom):
                tm_idx_list.append(atom.GetIdx())
        else:
            if _is_metal(atom):
                tm_idx_list.append(atom.GetIdx())
    return tm_idx_list, [outmol.GetAtomWithIdx(n) for n in tm_idx_list]

def _is_metal(atom):
    """
        Metal atomic numbers are hard-coded.
    """
    atom_num = atom.GetAtomicNum()
    ## First two rows: Li, Be, Na, Mg, Al
    return (atom_num in [3, 4, 11, 12, 13]) or \
    (19 <= atom_num <= 31) or \
    (37 <= atom_num <= 50) or \
    (55 <= atom_num <= 83) or \
    (87 <= atom_num <= 104) # up to 104 element
    
def _is_transition_metal(atom):
    """
        Transition metal atomic numbers.
    """
    atom_num = atom.GetAtomicNum()
    return (21 <= atom_num <= 30) or \
    (39 <= atom_num <= 50) or \
    (72 <= atom_num <= 80) # up to the 3rd row

def remove_isolated_atoms(mol):
    """
        Note that RemoveAtom() function can only delete one atom at a time.
        Atom indices change after each removal.

        To address this, sort the indices and delete from the largest index, following the method here:
            https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/2017062518163195443411@gmail.com/
        --> because only higher indices will change after removal
    """
    emol = EditableMol(mol)
    isolated_indices = [atom.GetIdx() for atom in mol.GetAtoms() if len(atom.GetNeighbors())<1]
    sorted_indices = sorted(isolated_indices, reverse=True)
    for idx in sorted_indices:
        emol.RemoveAtom(idx)
    new = emol.GetMol()
    return new

def count_coord_atoms(mol):
    """
        Coordinating atoms are connected to transition metals by
        dative bonds, so we only need to count the number of dative
        bonds.

        Note: sometimes, generative models (either LLMs or CLMs)
            may produce SMILES with WRONG extra dative bonds
            (e.g., replace a single bond with a dative bond).
            To address this, we need to fix all redundant dative
            bonds before counting them for coordinating atoms.
    """
    num = 0
    for bond in mol.GetBonds():
        if bond.GetBondType()==Chem.BondType.DATIVE:
            num += 1
    return num

def get_all_ring_sizes(mol):
    """
        RDKit has an object class (RingInfo) which contains all
        information of rings, including atoms and bonds in rings,
        whether rings are fused with other rings, and ring sizes.
    """
    ri = mol.GetRingInfo()
    return [len(n) for n in ri.AtomRings()]

def get_fused_ring_idx(mol):
    """
        Fused ring describes the complexity of molecules
    """
    ri = mol.GetRingInfo()
    return [idx for idx in range(len(ri.AtomRings())) if ri.IsRingFused(idx)]

"""
    Copyright ©2025  The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 408, Berkeley, CA 94704-1362, otl@berkeley.edu.

    Created by John Smith and Mary Doe, Department of Statistics, University of California, Berkeley.

    IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS
"""