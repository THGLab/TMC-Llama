"""
    A collection of functions to identify ring digits (single-
    or double- digits) in SMILES. There are two key challenges
    to identify digits counting rings in SMILES:
        (1) Exclude the digits that are unrelated to rings
        (2) Identify double digits which count rings
    Details will be discussed in the following documentation.
    
    Overall, all functions here are text-based operations
    unrelated to RDKit, but the algorithms in these functions
    are based on SMILES (or TMC-SMILES) syntax notation system
    and a few of them are specifically for RDKit.
"""

import numpy as np
from itertools import islice, product

"""
    Section 1: identify ring digits in SMILES

    In SMILES, rings begin and end with a pair of identical
    digits. These digits represent ring identities and can be
    reused after ring closure.
        e.g., c1ccccc1Cc1ccncc1 equals c1ccccc1Cc2ccncc2
        -> Note: this also implies that the real number of rings
           in a SMILES can be more than the largest identity
           digit.
    
    If other rings begin / end at the same position simutaneously,
    a new identity digit will be assigned to distinguish these
    rings.
        e.g., [FeH+]<-1234<-[CH2-]C->1=C->2C->3=C->4C
                      ||||
            (four rings)

    Having larger ring digits means that many rings occur at the
    same place, which implies fused rings and highly complicated
    ring systems. If the total number of rings is ten or more,
    RDKit usually adopts a syntax notation which uses '%' to track
    rings with double digits (e.g., c%11ccccc%11)
    
        -> Ref.: https://www.rdkit.org/docs/RDKit_Book.html#ring-closures
        -> Note: RDKit also adopts %(N) notation which includes
            brackets to track ring labels with more than two
            digits
                e.g., C%(102)CCCCC%(102)
                
            The bracket notation IS NOT CONSIDERED in this script
            because such SMILES are sooooooooooooo rare and probably
            never occur if using canonical SMILES.
            
            However, this may remain vulnerable and potentially
            a bug. Future algorithms may want to fix this. 
    
"""

def locate_digits(smi):
    """
        The fundamental idea is to locate all locations
        (indices) of ring digits and randomly remove the digits
        which have an odd number of occurrences.
            e.g., {"1": [i1, i2, ..., i9], "2": [...], ...}
                    |-- ring digits
                         |-- indices represent where the digits
                             occur
                                      |-- if the number is odd,
                                          remove one since an
                                          odd occurrence means
                                          incomplete rings
        The purpose is to:
            i. identify all digits in a SMILES string
            ii. exclude digits that are unrelated to rings
            iii. build a dictionary of ring digits and their
            locations
            iv. remove one for each digit which has an odd
            number of occurrence
            v. test if the output SMILES is valid
                -> Note: removing rings (in a brutal-force
                    manner) may cause kekulization issues,
                    valence issues, and aromatic atoms outside
                    the ring, so that the SMILES strings are
                    still invalid

        The difficult step is (ii), which is also largely
        empirical & text-engineering. Better (ii) will lead to
        higher conversion rates.
    """
    digits = [str(n) for n in range(1, 10)]
    digit_dict = {}
    for idx, char in enumerate(smi):
        if char in digits:
            if not int(char) in digit_dict.keys():
                digit_dict[int(char)] = [idx]
            else:
                digit_dict[int(char)].append(idx)
    return digit_dict

def locate_double_digit_rings(smi):
    """
        In SMILES, double-digit rings always begin with a '%'.
        Therefore, in this script, we assign the indices of
        double-digit labels as the indices of %.

        The good thing about '%'-based double digits of rings
        is, all double digits begin with '%' are ring identities.
        In contrast, single digits or multiple single digits
        may be hydrogen atom numbers, oxidation states, or
        something else.
            -> so double-digit ring identities don't need
                screening
            -> but single-digit ring identities need to be
                screened from double-digits:
                    e.g., c1cc%12ccc1CCC%12
                               |-- the single digit '1' in '12'
                                   needs to be screened from
                                   the locations of '1's

        NOTE: here we take the indices of '%' as locations of
        double-digit ring identities, not directly the locations
        of ring identities.
    """
    all_indices = [idx for idx, value in enumerate(smi) if value=='%']
    distinct_rings = {}
    for idx in all_indices:
        ring_num = int(smi[idx+1:idx+3])
        if not ring_num in distinct_rings.keys():
            distinct_rings[ring_num] = [idx]
        else:
            distinct_rings[ring_num].append(idx)
    return distinct_rings

"""
    Section 2: screen unrelated digits

    Many digits in SMILES are unrelated to rings and we need to
    screen them. The efficiency of programs fixing SMILES with
    unclosed rings depends on how accurate the screening algorithms
    are. In fact, even a minimal removal algorithm without any
    screening function will work, but with very low efficiency
    retrieving SMILES with unclosed rings.
    
    - Double-digit ring identities don't need screening because
        they're coupled with specific '%' signs.
    - Single-digit ring identities needs to be screened because
        the algorithm to locate these digits in strings are too
        simple (returning locations of all digits):
        
        i. screen_double_digits: screen indices of 'SINGLE-DIGITS'
            if they are double counted in double-digit identities.
                e.g., screen the '2' in '%12'
                
        ii. _check_hydrogen_and_positive: screen digits related
            to hydrogen counts & positive charges

        iii. screen_digits: screen all unrelated digits while also
            having an embeded algorithm for negative charges
            (which is missing in ii.)

    Overall, the algorithms in this section are developed in a 
    'case-by-case' manner through observations. Details are
    discussed in each function below.
"""
def screen_double_digits(smi, digit_dict, double_dict):
    """
        Screen double-counted '2' from '%12'
            e.g., {"2": [0, 6, 17, 25, 40, 48], "12": [15, 23],
                   "21": [39, 47]}
                                |   |-- these two '2' are from "%12"
                                (15 is the index of '%', and then
                                 '1' and '2' are 16 and 17)
                                        |   |
                                        |   |-- these two are from
                                                "%21"
        
        Attri:
            digit_dict: dictionary of single-digit indices
            double_dict: dictionary of double-digit indices
            -> Note: NO OVERLAP between digit_dict & double_dict!
    """
    screened_dict = {} ## final dictionary for single-digits

    ## Screen each single-digit identity
    for key, indices in digit_dict.items():
        ## Step 1: detect double-counted digits
        ## e.g., '2' is NOT in '%10'
        
        ## key: single digits
        ## double: double digits
        filters = [double for double in double_dict.keys() if str(key) in str(double)]
        if len(filters)<1:
            screened_dict[key] = indices
        else:
            screened_indices = []
            all_double_indices = [n for k in filters for n in double_dict[k]]
            for idx in indices:
                """
                    Step 2: remove the indices that are double
                    counted in double-digit identities.
                    Note that double-digit indices are for '%'!
                """
                check = 0
                for didx in all_double_indices:
                    ## check if this digit occurs in two chars after '%'
                    ## -> <index of '%'> +1 or +2
                    if ((idx-didx)>0) and ((idx-didx)<3):
                        check = 1 ## this index needs to be removed
                        break ## no need for further loop
                if not check:
                    screened_indices.append(idx)
            screened_dict[key] = screened_indices
    return screened_dict

def screen_digits(smi, digit_dict):
    """
        Return only single digits and their indices for ring
        identities

        Single-digit rings are complicated:
        Digits can be found:
            1. as charges: Co+1 --> after +
                           Co-1 --> after -
            2. as double-digit charges:
                Co+10 --> so we need to check at least 2 chars
                    before, looking for '+' and '-'.

                HOWEVER, there are exceptions for fake 'negative
                charges' which begin with a '-':
                
                    2.1: '->1' & '<-1' are still valid ring
                        identities which have '-' signs
                    2.2: occasionally, there are strings like
                        '-c1ccccc1' with '-' but they are valid
                        ring digits as well.
                        
            3. as hydrogen counts:
                [CH2+] --> behind 'H'
                
        Note: I can hardly think of any type of ions having triple-
            digit charges like +100, or any atom having more than
            99 hydrogens like CH101. If any model produces something
            like these, let it fail.
            --> IT DESERVES FAILURE!!! ( ˶°ㅁ°) !!

        Summary:
            i. screen the single digits which are double counted
                in double-digit ring identities
                
            ii. screen the single digits that count hydrogen (after
                'H'), positive charges (after '+'), and negative
                charges (after '-')
            
            iii. in (ii), keep the exceptions of fake 'negative
                charges', including [BUT NOT LIMITED TO!]
                '->1', '<-1', and '-c1'

            iv. double-digit charges may occur so check at least
                two chars before the identified single-digits

            v. Note: (iii) and (iv) are vulnerable because (iii)
                may not be a complete list of exceptions (only
                what I have observed), and (iv) doesn't consider
                triple-digit charges or more.
                (BUT I DON'T BELIEVE THERE CAN BE MORE...)
    """
    screened_dict = {}
    for key, indices in digit_dict.items():
        screened_indices = []
        for idx in indices:
            if idx<=2:
                ## must be a ring representation because it's
                ## the start of a string
                screened_indices.append(idx)
            else:
                ## consider two chars before
                chars = smi[idx-2:idx]
                ## Step 1: filter digits for positive charges and
                ## hydrogens
                if _check_hydrogen_and_positive(chars):
                    ## must be a hydrogen count or a charge
                    ## so not appending the index to the final
                    ## screened list
                    continue
                
                ## Step 2: filter digits for negative charges
                elif '-' in chars:
                    reduced = chars.index('-')
                    negative_idx = idx-2+reduced
                    if smi[negative_idx-1]=='<':
                        ## the case of '<-x3', valid for rings
                        screened_indices.append(idx)
                    else:
                        ## Step 3: decide if it's a fused ring like
                        ## -c3ccccc3
                        if reduced==0:
                            if chars[1].isdigit():
                                ## the case of '3' in '-13'
                                ## must be a negative charge
                                continue
                            else:
                                """
                                    since the 'x' in '-x1' is not
                                    a digit, it's very likely to
                                    be a letter for chemical
                                    elements, valid for rings
                                """
                                screened_indices.append(idx)
                        else:
                            """
                                Very likely to be a negative charge.
                                Note:
                                The case of '<-1' or other cases may
                                be missed, but the current efficiency
                                is good enough to retrieve ~ 60% of
                                the unclosed strings
                            """
                            continue
                else:
                    screened_indices.append(idx)
        screened_dict[key] = screened_indices
    return screened_dict

def _check_hydrogen_and_positive(chars):
    """
        This algorithm checks (at most) double-digit counts and
        charges, like 'CH10' (as hydrogen counts) & 'Co+10'
        (as positive charges)
        
        Note:
            The balance here is the checked substrings should be
            short to reduce algorithm complexity, while multi-
            digit charges require longer substrings to detect.
            -> No perfect algorithm, engineering to catch as many
                exceptions as possible.
        Note:
            Negative charges should be handled separately
    """
    if ('H' in chars) or ('+' in chars):
        if chars[0] in ['+', 'H']:
            """
                Observed exceptions here:
                [nH]1cc... & [Co+]1...
                both are valid ring identities

                There may be other exceptions too...
            """
            if chars[1]==']':
                ## Not for hydrogen counts or positive charges
                return False 
        ## digits are for hydrogens and positive charges
        ## invalid for rings
        return True
    else:
        return False

"""
    Section 3: remove the identified redundant ring digits
"""
def remove_fixed_slices(smi, indices, l=1):
    """
        To remove substrings (redundant digits)
        Assuming all substrings have identical lengths

        Note: when removing substrings, the indices of
            remaining substrings will change if they are
            behind the removed one.
        --> Solution:
            Sort the substring indices, build a new string
            from the remaining of the previous string:
                e.g., (Previous)
                      ThisIsAStringForDemonstration.
                            |||||||       |||||||||-> removed
                      0     6      13     20       29
                      
                      (Adding)
                      ThisIs   +   ForDemo     +   .
                      prev[0:6]+prev[13:20]+prev[29:]
                      indices of removed strings: [6, 20]
                      
                      (Final)
                      --> ThisIsForDemo.
    """
    sorted_indices = sorted(indices)
    new_smi = smi[:sorted_indices[0]]
    for iidx, idx in enumerate(sorted_indices[1:]): ## <- from 1!
        """
            Using the above example:
                removed = [6, 20]
                           |-- removed[0]
                              |-- removed[1]
            To build the new string, the new string should be
            initialized from prev_string[:removed[0]]. Then,
            adding the remaining substrings (the positions of
            removed substrings should be sorted first) from
            THE FIRST element (NOT the zero-th) of the removed
            list:
                [(prev_index_in_removed)+l:(current_index_in_removed)]
        """
        new_smi += smi[(sorted_indices[iidx]+l):idx]
    new_smi += smi[sorted_indices[-1]+l:]
    return new_smi

def remove_slices(smi, indices, lengths):
    """
        Removing substrings which have various lengths
    """
    slices = [(idx, l) for idx, l in zip(indices, lengths)]
    sorted_slices = sorted(slices, key=lambda k: k[0])
    sorted_indices = [n[0] for n in sorted_slices]
    sorted_lengths = [n[1] for n in sorted_slices]
    new_smi = smi[:sorted_indices[0]]
    for iidx, idx in enumerate(sorted_indices[1:]): ## <- from 1!!!
        prev_idx = sorted_indices[iidx]
        prev_l = sorted_lengths[iidx]
        new_smi += smi[(prev_idx+prev_l):idx]
    new_smi += smi[sorted_indices[-1]+sorted_lengths[-1]:]
    return new_smi

"""
    Section 4: summary
"""
def fix_simple_smi(smi, max_attempts=10000):
    """
        Assuming we only have single-digit rings
    """
    all_digits = locate_digits(smi)
    screened_digits = screen_digits(smi, all_digits)
    
    ## Looking for digits that have odd number of occurrence
    removable_digits = {k: v for k, v in screened_digits.items() if len(v)%2==1}
    ## All possible combinations of removed indices
    possibles = np.prod(np.array([len(v) for k, v in removable_digits.items()]))
    if possibles>max_attempts:
        print('Warning! Exceeding maximum number of variations!')
    
    ## Provide one index to remove from SMILES for each ring digit
    ## --> limited by maximum attempts
    attempts = []
    data = [(k, v) for k, v in removable_digits.items()]
    matrix = [n[1] for n in data]
    limited_combinations = list(islice(product(*matrix), max_attempts))
    for combo in limited_combinations:
        attempts.append(remove_fixed_slices(smi, combo, l=1))
    return attempts

def fix_complex_smi(smi, max_attempts=10000):
    """
        Now we have double-digit rings
    """
    all_digits = locate_digits(smi) ## from 1 to 9
    double_digits = locate_double_digit_rings(smi)
    
    ## exclude indices from H, and charges
    screened_digits = screen_digits(smi, all_digits)
    ## exclude double-counted indices from double-digit rings
    pure_single_digits = screen_double_digits(smi, screened_digits, double_digits)
    
    removable_digits = {k: v for k, v in double_digits.items() if len(v)%2==1}
    for k, v in pure_single_digits.items():
        if len(v)%2==1:
            removable_digits[k] = v
    possibles = np.prod(np.array([len(v) for k, v in removable_digits.items()]))
    if possibles>max_attempts:
        print('Warning! Exceeding maximum number of variations!')
    
    attempts = []
    data = [(k, v) for k, v in removable_digits.items()]
    lengths = []
    for d in data:
        if d[0]<10:
            ## single-digit rings: remove one char
            lengths.append(1)
        else:
            ## double-digit rings: remove three chars (%12)
            lengths.append(3)
    matrix = [n[1] for n in data]
    limited_combinations = list(islice(product(*matrix), max_attempts))
    for combo in limited_combinations:
        attempts.append(remove_slices(smi, combo, lengths))
    return attempts

"""
    Copyright ©2025  The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 408, Berkeley, CA 94704-1362, otl@berkeley.edu.

    Created by John Smith and Mary Doe, Department of Statistics, University of California, Berkeley.

    IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS
"""