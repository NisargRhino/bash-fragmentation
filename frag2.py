import os
import shutil
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors, SanitizeMol
from tqdm import tqdm
import pandas as pd
from dockstring import load_target
import argparse
import random

def is_disconnected(mol):
    """Checks if a molecule is disconnected."""
    return len(Chem.GetMolFrags(mol)) > 1

def filter_fragments(fragments, parent_mol, similarity_threshold=0.5, min_atoms=5, max_atoms=30):
    """Filters fragments based on similarity to parent molecule, atom count, and other criteria."""
    filtered_fragments = []
    parent_mol = Chem.MolFromSmiles(parent_mol)

    for frag_smiles in fragments:
        frag_mol = Chem.MolFromSmiles(frag_smiles)
        if frag_mol is None:
            continue
        similarity = calculate_similarity(parent_mol, frag_mol)
        num_atoms = frag_mol.GetNumAtoms()
        if similarity >= similarity_threshold and min_atoms <= num_atoms <= max_atoms and not is_disconnected(frag_mol):
            filtered_fragments.append(frag_smiles)

    return filtered_fragments

def targeted_fragmentation(smiles, max_fragments=100):
    """Splits a molecule into fragments by systematically removing atoms from the periphery.

    Args:
        smiles: The SMILES string of the molecule.
        max_fragments: Maximum number of fragments to generate.

    Returns:
        A list of SMILES strings representing the fragments.
    """

    mol = Chem.MolFromSmiles(smiles)

    # Identify potential bond breaking points
    bonds_to_break = []
    for bond in mol.GetBonds():
        bond_order = bond.GetBondTypeAsDouble()
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        atom1_type = atom1.GetAtomicNum()
        atom2_type = atom2.GetAtomicNum()
        # Adjust bond breaking criteria based on your molecule type and desired fragments
        if bond_order == 1 and (atom1_type in [6, 7, 8, 9] or atom2_type in [6, 7, 8, 9]):
            bonds_to_break.append(bond)

    fragments = []
    for bond in bonds_to_break:
        new_mol = Chem.RWMol(mol)
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        new_mol.RemoveBond(atom1_idx, atom2_idx)
        new_mol = new_mol.GetMol()

        try:
            Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
            if not is_disconnected(new_mol):
                fragments.append(Chem.MolToSmiles(new_mol))
        except:
            pass  # Handle invalid molecules

        if len(fragments) >= max_fragments:
            break

    return fragments
def dock_fragments(fragments, target_name, docking_dir, mol2_path, center_coords, box_sizes):
    os.makedirs(docking_dir, exist_ok=True)

    # Convert mol2 to pdbqt
    convert_command = f"obabel -imol2 {mol2_path} -opdbqt -O {os.path.join(docking_dir, target_name + '_target.pdbqt')} -xr"
    print(f"Running command: {convert_command}")
    os.system(convert_command)

    conf_path = os.path.join(docking_dir, target_name + '_conf.txt')
    with open(conf_path, 'w') as f:
        f.write(f"""center_x = {center_coords[0]}
center_y = {center_coords[1]}
center_z = {center_coords[2]}

size_x = {box_sizes[0]}
size_y = {box_sizes[1]}
size_z = {box_sizes[2]}""")

    target = load_target(target_name, targets_dir=docking_dir)

    best_scores = []
    fragments_with_scores = []

    for frag in tqdm(fragments):
        try:
            mol = Chem.MolFromSmiles(frag)
            if mol is not None:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
                if not is_disconnected(mol):
                    score, __ = target.dock(frag)
                    best_scores.append(score)
                    fragments_with_scores.append(frag)
                else:
                    print(f"Fragment {frag} is disconnected")
            else:
                print(f"Invalid fragment: {frag}")
        except Exception as e:
            print(f"Error docking fragment {frag}: {e}")
            # Add more detailed error handling or debugging here
            best_scores.append(None)

    return fragments_with_scores, best_scores

def main(input_smiles, mol2_path, docking_dir, target_name, center_coords, box_sizes, output_path):
    fragments = targeted_fragmentation(input_smiles)

    if not fragments:
        print("No fragments were generated.")
        return

    print(f"Generated {len(fragments)} fragments.")

    fragments_with_scores, docking_results = dock_fragments(fragments, target_name, docking_dir, mol2_path,
                                                            center_coords, box_sizes)

    if not any(docking_results):
        print("No successful docking results.")
        return

    results_df = pd.DataFrame({'Fragment': fragments_with_scores, 'DockingScore': docking_results})
    results_df = results_df.dropna().sort_values(by='DockingScore')

    if results_df.empty:
        print("No successful docking results after filtering.")
        return

    best_fragment = results_df.iloc[0]

    print(f"Best fragment: {best_fragment['Fragment']} with score: {best_fragment['DockingScore']}")

    results_df.to_csv(output_path, index=False)

    return best_fragment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drug fragmentation and docking script')

    parser.add_argument('--input_smiles', type=str, required=True, help='SMILES string of the input molecule')
    parser.add_argument('--mol2_path', type=str, required=True, help='Path to mol2 file')
    parser.add_argument('--docking_dir', type=str, default='dockdir', help='Docking directory name/path')
    parser.add_argument('--target_name', type=str, default='target', help='Target name')
    parser.add_argument('--center_coords', type=float, nargs=3, help='Center coordinates for docking box (X Y Z)')
    parser.add_argument('--box_sizes', type=float, nargs=3, help='Box sizes for docking (X Y Z)')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the docked results CSV')

    args = parser.parse_args()

    main(args.input_smiles, args.mol2_path, args.docking_dir, args.target_name, args.center_coords, args.box_sizes,
         args.output_path)
