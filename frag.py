import os
import shutil
from rdkit import Chem
from rdkit.Chem import BRICS
from tqdm import tqdm
import pandas as pd
from dockstring import load_target
import argparse


#def canonicalize(smiles):
#    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)


def fragment_molecule_brics(smiles):
    mol = Chem.MolFromSmiles(smiles)
    frags = BRICS.BRICSDecompose(mol)
    print("fragments are : " + str(frags))
    return list(frags)


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
            score, __ = target.dock(frag)
            best_scores.append(score)
            fragments_with_scores.append(frag)
        except Exception as e:
            print(f"Error docking fragment {frag}: {e}")
            best_scores.append(None)

    return fragments_with_scores, best_scores


def main(input_smiles, mol2_path, docking_dir, target_name, center_coords, box_sizes, output_path):
    fragments = fragment_molecule_brics(input_smiles)



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
