"""
SMILES parsing and error-message handling using RDKit.

This module captures RDKit's C++ stderr (syntax and chemical validation messages)
so they can be inspected in Python. It provides:
- Parsability check: whether a SMILES string can be read and sanitized.
- Helpers to append captured errors/warnings into dictionary-based tables.
"""

from io import StringIO
from contextlib import redirect_stderr

from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolToSmiles
from rdkit.Chem import SanitizeMol, MolFromSmiles, DetectChemistryProblems


def _catch_op_messages(smi=None, mol=None, read_smiles=False, sanitize=False):
    """
    Run RDKit operations while capturing C++ stderr and chemistry problems.

    Used for:
    1. Reading SMILES without sanitization → surfaces syntax errors.
    2. Sanitizing (read+sanitize, or sanitize an existing mol) → surfaces chemical errors.

    Parameters
    ----------
    smi : str or None
        SMILES string. Required if read_smiles is True.
    mol : Mol or None
        RDKit molecule. Used only when sanitize is True and read_smiles is False.
    read_smiles : bool
        If True, parse `smi` with MolFromSmiles (with or without sanitization).
    sanitize : bool
        If True, run full sanitization (either on parsed SMILES or on `mol`).

    Returns
    -------
    output : Mol or None
        The resulting molecule, or None if parsing/sanitization failed.
    message_dict : dict
        Keys: 'Errors' (exception text), 'Warnings' (stderr), 'Problems' (list of
        RDKit problem messages).
    """
    if read_smiles and smi is None:
        raise ValueError("No SMILES provided.")
    if (smi is None and mol is None) or (not read_smiles and sanitize and mol is None):
        raise ValueError("Nothing to do: need smi (with read_smiles) or mol (with sanitize).")

    error = ""
    warning = ""
    total_probs = []

    try:
        with StringIO() as buf:
            with redirect_stderr(buf):
                if read_smiles and not sanitize:
                    output = MolFromSmiles(smi, sanitize=False)
                    if output is not None:
                        probs = DetectChemistryProblems(output)
                        for prob in probs:
                            total_probs.append(prob.Message())
                elif read_smiles and sanitize:
                    output = MolFromSmiles(smi)
                elif not read_smiles and sanitize:
                    output = Mol(mol)
                    SanitizeMol(output)
                else:
                    output = None
                warning = buf.getvalue()
    except Exception as e:
        output = None
        error = str(e)

    return output, {"Errors": error, "Warnings": warning, "Problems": total_probs}


def get_parsability(smi):
    """
    Determine whether a SMILES string is parsable and sanitizable by RDKit.

    First parses without sanitization to detect syntax issues; if that succeeds
    and there are no chemistry problems, then parses with sanitization. This
    separates syntax errors from chemical errors (e.g., valence, kekulization).

    Parameters
    ----------
    smi : str
        SMILES string to check.

    Returns
    -------
    mol : Mol or None
        RDKit molecule if parsable and sanitizable, else None.
    parsable : bool
        True if the SMILES was fully parsed and sanitized.
    messages : dict
        Keys: 'Errors', 'Warnings', 'Problems'. Contains exception text, stderr,
        and RDKit problem messages for diagnostics.
    """
    unsanitized_mol, unsanitized_messages = _catch_op_messages(
        smi=smi, read_smiles=True, sanitize=False
    )
    if unsanitized_mol is not None and len(unsanitized_messages["Problems"]) < 1:
        output, messages = _catch_op_messages(smi=smi, read_smiles=True, sanitize=True)
        if output is not None:
            return output, True, messages
        messages["Errors"] += (
            "Passing sanitization check strangely!\n" + unsanitized_messages["Errors"]
        )
        messages["Warnings"] += "------------------------\n" + unsanitized_messages["Warnings"]
        return output, False, messages
    return None, False, unsanitized_messages


def append_error_messages(message_dict, error_dict, subtitle):
    """
    Append one row of error/warning/problem fields into existing list columns.

    Used to build DataFrame columns such as Parse_errors, Parse_warns, Parse_probs
    from the dict returned by get_parsability (or _catch_op_messages).

    Parameters
    ----------
    message_dict : dict of list
        Dict that will get keys f'{subtitle}_errors', f'{subtitle}_warns',
        f'{subtitle}_probs'. Values are lists to which one entry is appended.
    error_dict : dict
        Must have keys 'Errors', 'Warnings', 'Problems'. 'Problems' should be
        a list of strings (e.g., RDKit problem messages).
    subtitle : str
        Prefix for the keys added to message_dict (e.g. 'Parse').
    """
    message_dict[f"{subtitle}_errors"].append(error_dict["Errors"])
    message_dict[f"{subtitle}_warns"].append(error_dict["Warnings"])
    message_dict[f"{subtitle}_probs"].append("//".join(error_dict["Problems"]))
