import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Mol
from rdkit.Chem import SanitizeMol

"""
    This script aims to establish functions to fix atoms
    with valence errors, e.g., nitrogen atoms with four
    single bonds.

    Valence errors are common issues in generated SMILES
    strings, strings that are retrieved from unclosed rings,
    and, especially, strings that have extra dative bonds.

    The algorithm is to replace the atoms having valence
    errors by atoms that permit higher valences.
        e.g., CN(C)(C)C --> CC(C)(C)C

    Several features for algorithms in this script:
        - molecular properties like ring structures and
          spatial complexity remain the same
          
        - functional groups and chemical properties are
          changed, so it's plausible if the 'fixed'
          molecules are chemically valid
          
        - the replacement atoms are pre-defined and we
          use these atoms based on their permitted valences.
          Thus, these choices are plausible chemically
          
        - The most challenging situation is to have charged
          atoms with valence errors. Since both charges and
          bond orders are considered explicit valences, it's
          undecided if the replacement atoms should be 
          selected according to their total valences or the
          bond orders. Details will be discussed below.
          
"""

def fix_dative_bonds(mol, use_transition_metal=False):
    """
        This function identifies and fixes the unnecessary
        dative bonds in molecules.
        
        In TMC-SMILES, only bonds which connect TM atoms
        are dative, while string-based generative models
        occasionally create random dative bonds in molecules.
        These dative bonds should be fixed back to single
        bonds at least.

        To identify unnecessary dative bonds:
            At least one of the begin and end atoms is a
            metal or TM atom.
        To fix the unnecessary dative bonds:
            Change identified dative bonds to single bonds.
        -> Note:
            Changing dative bonds to single bonds may raise
            several problems:
            - dative bonds are not counted in RDKit while
                single bonds have bond order of one. Changing
                dative bonds to single bonds may cause valence
                errors for some atoms
            - there's no evidence that single bonds are the
                best or even the appropriate choice for
                different chemical environments
    """
    outmol = Mol(mol)
    bond_errors = []
    for bond in outmol.GetBonds():
        if bond.GetBondType()==Chem.BondType.DATIVE:
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            begin_atom = outmol.GetAtomWithIdx(begin_atom_idx)
            end_atom = outmol.GetAtomWithIdx(end_atom_idx)
            
            if use_transition_metal:
                check_begin = _is_transition_metal(begin_atom)
                check_end = _is_transition_metal(end_atom)
            else:
                check_begin = _is_metal(begin_atom)
                check_end = _is_metal(end_atom)
            
            ## at least one of the atoms is metal/TM
            if (not check_begin) and (not check_end):
                bond.SetBondType(Chem.BondType.SINGLE)
                bond_errors.append(bond.GetIdx())
    return outmol, bond_errors

def _is_metal(atom):
    """
        _is_metal & _is_transition_metal are identical with
        with those in molecular_properties.py
    """
    atom_num = atom.GetAtomicNum()
    return (atom_num in [3, 4, 11, 12, 13]) or \
    (19 <= atom_num <= 31) or \
    (37 <= atom_num <= 50) or \
    (55 <= atom_num <= 83) or \
    (87 <= atom_num <= 104)
    
def _is_transition_metal(atom):
    atom_num = atom.GetAtomicNum()
    return (21 <= atom_num <= 30) or \
    (39 <= atom_num <= 50) or \
    (72 <= atom_num <= 80)

def GetTotalBondOrders(atom):
    """
        Count the total bond order of central atom
        (usually the atom with valence error)
        Dative and aromatic bonds are excluded.

        Returns:
            - total bond order (double)
            - if there are aromatic bonds (bool)

        Note: aromatic bonds are treated as exceptions in
        this script and workflow. Atoms having valence
        errors with aromatic bonds will NOT be fixed.
    """
    bo = 0
    for bond in atom.GetBonds():
        if bond.GetBondType()==Chem.rdchem.BondType.AROMATIC:
            ## raise ValueError('Encountering aromatic bond!', atom.GetSymbol())
            print('Encountering aromatic bond!', atom.GetSymbol())
            return 0, True
        if bond.GetBondType()!=Chem.rdchem.BondType.DATIVE:
            bo += bond.GetBondTypeAsDouble()
    return bo, False

def GetImpliedValence(atom):
    """
        Count the implied valence of atoms with errors
        (should be higher than permitted valence)
    """
    bond_orders, aromatic = GetTotalBondOrders(atom)
    if aromatic:
        return 0, aromatic
    charge = atom.GetFormalCharge()
    total_valence = bond_orders - charge
    return total_valence, False
    
def AutoCorrectAtom(mol, atomId, atomSymbol=None, sanitize=False):
    """
        Replacing atoms with valence errors by a dictionary
        of pre-defined atoms. The replacement atoms are
        common in organic chemistry having various
        perimitted valences, ranging from 1 to 6.

        The most challenging part here is to fix the charged
        atoms with valence errors. For instance, having
        a negatively charged oxygen with three single
        bonds,
        
              - C[O-](C)C
                the oxygen total valence is 4, while the
                permitted valence is 2

              - if we fix the oxygen according to its
                explict valence 4, we should use a carbon
              --> C[C-](C)C (negatively charged)
              --> C[C](C)C (unpaired electron) -- RDKit
              --> CC(C)C
                {Note}: interestingly, RDKit algorithm will
                set the replaced carbon with an unpaired
                electron (the 2nd output) if you sanitize
                the molecule

              - if we fix the oxygen using its bond orders
                and a neutral replacement atom (I prefer
                this approach myself), we should use a
                nitrogen
              --> CN(C)C
        
        This algorithm adjusts the atom type using explicit
        valences:
            Explicit valence = total bond orders + charges
            - Note: negative charges are positive valences
            - Note: total bond orders = sum(bond as double
                    integers)

        Attri:
            - atomId: the index of atom with valence error
                in the molecule
            - atomSymbol: used to confirm that the given
                atom with error is correct (the atom type
                should match)

        Returns:
            - fixed molecule
            - aromatic (bool): if the atom with valence
                errors has aromatic bonds, the molecule
                will be tossed as an exception
    """
    ## A pre-defined dictionary of replacement atoms: their
    ##    atomic numbers, symbols, and max permitted valences
    repAtomNumdict = {6: {'Symbol': 'C', 'Valence': 4},
                      7: {'Symbol': 'N', 'Valence': 3},
                      8: {'Symbol': 'O', 'Valence': 2},
                      9: {'Symbol': 'F', 'Valence': 1},
                      15: {'Symbol': 'P', 'Valence': 5},
                      16: {'Symbol': 'S', 'Valence': 6}}
    ## The valence dictionary
    maxVdict = {v['Symbol']: v['Valence'] for k, v in repAtomNumdict.items()}
    
    ## The atomic number dictionary to set atom types in RDKit
    v2AtomNum = {v['Valence']: k for k, v in repAtomNumdict.items()}
    
    new_mol = Mol(mol)
    atom = mol.GetAtomWithIdx(atomId)
    
    ## Confirm that the given atom is correct
    symb = atom.GetSymbol()
    if atomSymbol!=None:
        if atomSymbol!=symb:
            raise ValueError('Atom type not matching!', atomSymbol, 'vs. real-', symb)
    total_v, aromatic = GetImpliedValence(atom)
    
    ## Atoms with aromatic bonds are exceptions
    if aromatic:
        return None, aromatic

    if not total_v in v2AtomNum.keys():
        raise ValueError('Replacing element not found!', total_v)
    
    """
        If the implied valence (total bond order + charges)
        is larger than the permitted valence, the output
        molecule may be a little bit weird after RDKit-
        sanitization,
            e.g., having a carbon with an unpaired electron.

        (Not sure if it's caused by RDKit. It's also difficult
        to change explicit & implicit valence in RDKit)

        Probably not a major issue because the output
        molecules are usable and sanitizable anyway.
    """
    if symb in maxVdict.keys():
        if total_v <= maxVdict[symb]:
            print('Warning: explicit valence lower than maximum. Are you sure of replacing it?')
    rep_atom_num = v2AtomNum[total_v]
    
    ## change the atom in new mol so it doesn't affect the
    ## original object
    new_atom = new_mol.GetAtomWithIdx(atomId)
    new_atom.SetAtomicNum(rep_atom_num)
    if atom.GetFormalCharge()!=0:
        new_atom.SetFormalCharge(0)
    
    ## May need sanitization!
    if sanitize:
        SanitizeMol(new_mol)
    return new_mol, False

def AutoDischargeAtom(mol, atomId, atomSymbol=None, discharge=True, sanitize=False):
    """
        An extension aglorithm for atom correction which
        handles charged atoms with valence errors. The algorithm
        prioritizes making the central atom charge neutral
        and then tries changing the atom type to fit the
        implied valence.
        
            e.g., the valence of nitrogen (4) in C[N-](C)C
            is higher than permitted (3)
            - in C[N-](C)C,
                total bond order = 3 (<= permitted 3)
                charge = -1
                --> total explicit valence = 4 (> 3)
            --> remove the charge CN(C)C
            
        If the implied valence after neutralizing the charge
        is still higher than permitted, reset the charge to
        0 first.
            e.g., O- with three single bonds
                C[O-](C)C --> CO(C)C --> CN(C)C
    """
    maxVdict = {'F': 1, 'O': 2, 'N': 3, 'C': 4, 'P': 5, 'S': 6}
    
    atom = mol.GetAtomWithIdx(atomId)
    symb = atom.GetSymbol()
    if atomSymbol!=None:
        if atomSymbol!=symb:
            raise ValueError('Atom type not matching!', atomSymbol, 'vs. real-', symb)
    
    # First check if the atom is known:
    if not symb in maxVdict.keys():
        print(f'Blind replacement: {symb}')
        return AutoCorrectAtom(mol, atomId, atomSymbol=atomSymbol, sanitize=sanitize)
        
    # Then check if the atom is charged.
    # if it's charged, try neutralizing the atom first
    charge = atom.GetFormalCharge()
    if charge!=0:
        bo, aromatic = GetTotalBondOrders(atom)
        if aromatic:
            return None, aromatic
        """
            Assuming the valence error is caused by unnecessary
            charges. This may work weirdly if the assumption
            doesn't apply.

            This assumption also implies that the atom is
            negatively charged because (bo - charge) < bo if
            charge > 0, and it's impossible that both
                bo <= max_valence
            and
                bo - charge > max_valence
            are true while charge > 0.
        """
        if bo<=maxVdict[symb] and (bo-charge)>maxVdict[symb]:
            new_mol = Mol(mol)
            new_atom = new_mol.GetAtomWithIdx(atomId)
            new_atom.SetFormalCharge(0)
            return new_mol, False
        
        """
            Assuming the total bond order is higher than the
            permitted valence while the atom is still charged.
            Then, the atom is unnecessarily charged.

            However, it remains unclear how to solve positively
            charged atoms with valence errors,
                e.g., a positively charged nitrogen with five
                single bonds, C[N+](C)(C)(C)C
                --> current algorithm will do this:
                    C[N+](C)(C)(C)C -> CN(C)(C)(C)C
                                    -> CP(C)(C)(C)C
                -> C[C+](C)(C)(C)C is also possible
        """
        if discharge:
            new_mol = Mol(mol)
            new_atom = new_mol.GetAtomWithIdx(atomId)
            new_atom.SetFormalCharge(0)
            return AutoCorrectAtom(new_mol, atomId, atomSymbol=atomSymbol, sanitize=sanitize)
    return AutoCorrectAtom(mol, atomId, atomSymbol=atomSymbol, sanitize=sanitize)

"""
    Copyright ©2025  The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 408, Berkeley, CA 94704-1362, otl@berkeley.edu.

    Created by John Smith and Mary Doe, Department of Statistics, University of California, Berkeley.

    IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS
"""