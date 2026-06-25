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
from rdkit.Chem import GetMolFrags
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

def find_tm_and_coordinating_atoms(mol, use_transition_metal=False):
    """
        Filtering TM atoms in molecules by their atomic numbers. Then,
        we find the neighboring atoms of the screened TM centers. The
        function will return the indices of TM centers and the indices
        of TM centers' neighboring atoms, which are coordinating atoms.
    """
    outmol = Mol(mol)
    tm_idx_list = []
    coord_atoms = []
    
    # Loop over atoms and find the TM(s)
    for atom in outmol.GetAtoms():
        if use_transition_metal:
            if _is_transition_metal(atom):
                tm_idx_list.append(atom.GetIdx())
        else:
            if _is_metal(atom):
                tm_idx_list.append(atom.GetIdx())
    if len(tm_idx_list)==0:
        raise ValueError('Missing metal atoms in the molecule.')
    elif len(tm_idx_list)>1:
        raise ValueError('Too many metal atoms in the molecule:', tm_idx_list)
    else:
        tm_idx = tm_idx_list[0]
    
    tm_atom = outmol.GetAtomWithIdx(tm_idx)
    for nn in tm_atom.GetNeighbors():
        coord_atoms.append(nn.GetIdx())
    return tm_idx, coord_atoms

def set_coordinating_atoms(mol, coord_idx_list, coord_name='is_coord_atom'):
    """
        A simple function to set atom properties for RDKit molecules.
    """
    outmol = Mol(mol)
    for atom in outmol.GetAtoms():
        idx = atom.GetIdx()
        if idx in coord_idx_list:
            atom.SetProp(coord_name, "1")
        else:
            atom.SetProp(coord_name, "0")
    return outmol

def update_coordinating_indices(mol, use_transition_metal=False):
    """
        Removing TM atoms from molecules will preserve the original atom
        ordering but the atoms with indices larger than TM atoms will
        decrease by the number of removed TM atoms
    """
    tm_idx, coord_atoms = find_tm_and_coordinating_atoms(mol, use_transition_metal=use_transition_metal)
    atom_map = {}
    idx = 0
    for atom in mol.GetAtoms():
        if atom.GetIdx()!=tm_idx:
            ## Here, the assumption is that the atom ordering will
            ## remain unchanged in EditableMol and only the atom 
            ## numberings after the TM will be changed
            ## --> removing TM will reduce atom indices by one
            atom_map[atom.GetIdx()] = idx
            idx += 1
    new_indices = [atom_map[i] for i in coord_atoms if i in atom_map.keys()]
    return new_indices, atom_map

def remove_tm(mol, use_transition_metal=False):
    """
        Removing TM atoms from molecules and updating the atom mapping.
        The dictionary contains atom mapping with format {old_index: new_index}
    """
    tm_idx, coord_atoms = find_tm_and_coordinating_atoms(mol, use_transition_metal=use_transition_metal)
    emol = EditableMol(mol)
    emol.RemoveAtom(tm_idx)
    new = emol.GetMol()
    new_coord_atoms, atom_map = update_coordinating_indices(mol, use_transition_metal=use_transition_metal)
    return new, new_coord_atoms, atom_map

def check_atom_prop(mol, prop_name="is_coord_atom"):
    for atom in mol.GetAtoms():
        propdict = atom.GetPropsAsDict()
        if not prop_name in propdict.keys():
            return False
    return True

def get_idx_by_prop(mol, prop_name):
    ## Assuming that the property values are binary (1 and 0)
    if check_atom_prop(mol, prop_name=prop_name):
        return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetProp(prop_name, autoConvert=True)]
    else:
        raise ValueError(f'Missing {prop_name} properties in the molecule!')

def molecule2frags_reindex(mol, coord_atoms=None, autoCheck=True, prop_name="is_coord_atom", warn=True):
    """
        The function takes RDKit molecule objects in which case the TM
        centers are removed, so the RDKit molecules here are isolated
        ligands, NOT complete TMC molecules.

        New atom mappings of TMC fragments, or isolated ligands, are
        included in output fragment lists too.
    """
    new_mol = Mol(mol)
    if (coord_atoms==None) and autoCheck:
        if warn:
            print('Detecting no list of coordinating atom indices. Try auto-generation')
        if check_atom_prop(new_mol, prop_name=prop_name):
            coord_list = get_idx_by_prop(new_mol, prop_name)
        else:
            raise ValueError('No coordinating atoms provided!')
    else:
        coord_list = [n for n in coord_atoms]
    
    frags_org_indices = []
    ## RDKit function: GetMolFrags
    frags = GetMolFrags(new_mol, asMols=True, fragsMolAtomMapping=frags_org_indices)
    # A list of tuples which includes fragments' mol obj and coordinating atoms' indices
    frag_output = []
    for frag, indices in zip(frags, frags_org_indices):
        ## Key: atom index in fragment molecule
        ## Value: atom index in original molecule
        frag_mapping = {atom.GetIdx(): indices[atom.GetIdx()] for atom in frag.GetAtoms()}
        ## Key: atom index in original molecule
        inverse_mapping = {v: k for k, v in frag_mapping.items()}
        
        ## indices in coord_list and mapping are all from the original molecule
        frag_coord_org = [n for n in indices if n in coord_list]
        frag_coord_reindex = [inverse_mapping[n] for n in frag_coord_org]
        frag_output.append((frag, frag_coord_reindex))
    return frag_output

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