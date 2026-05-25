from rdkit import Chem

def one_hot_encoding(value, choices):
    """Creates a one-hot vector based on a list of choices."""
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1 # 'Unknown' category
    return encoding

def get_atom_features(atom):
    """
    Extracts node features for an RDKit atom.
    Features: Atomic Number, Degree, Formal Charge, Hybridization, Aromaticity, Num Hs.
    """
    # 1. Atomic number (C, N, O, F, P, S, Cl, Br, I)
    atom_types =[6, 7, 8, 9, 15, 16, 17, 35, 53]
    features = one_hot_encoding(atom.GetAtomicNum(), atom_types)
    
    # 2. Degree (0, 1, 2, 3, 4, 5, 6)
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    
    # 3. Formal Charge (-1, 0, 1)
    features += one_hot_encoding(atom.GetFormalCharge(), [-1, 0, 1])
    
    # 4. Hybridization (SP, SP2, SP3)
    hyb_types =[Chem.rdchem.HybridizationType.SP, 
                 Chem.rdchem.HybridizationType.SP2, 
                 Chem.rdchem.HybridizationType.SP3]
    features += one_hot_encoding(atom.GetHybridization(), hyb_types)
    
    # 5. Aromaticity (Boolean -> 0 or 1)
    features.append(int(atom.GetIsAromatic()))
    
    # 6. Total Num Hs (0, 1, 2, 3, 4)
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    
    return features

def get_bond_features(bond):
    """
    Extracts edge features for an RDKit bond.
    Features: Bond Type, Conjugation, Ring membership.
    """
    # 1. Bond Type (SINGLE, DOUBLE, TRIPLE, AROMATIC)
    bond_types =[Chem.rdchem.BondType.SINGLE, 
                  Chem.rdchem.BondType.DOUBLE, 
                  Chem.rdchem.BondType.TRIPLE, 
                  Chem.rdchem.BondType.AROMATIC]
    features = one_hot_encoding(bond.GetBondType(), bond_types)
    
    # 2. Conjugation (Boolean)
    features.append(int(bond.GetIsConjugated()))
    
    # 3. In Ring (Boolean)
    features.append(int(bond.IsInRing()))
    
    return features
