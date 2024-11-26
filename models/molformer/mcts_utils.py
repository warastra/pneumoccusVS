
from chemprop.interpret import MCTSNode, extract_subgraph, find_clusters
from typing import List, Callable, Set, Dict
import torch
from rdkit import Chem
from transformers import AutoTokenizer

MolFormerXL_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

def mcts_rollout(node: MCTSNode,
                 state_map: Dict[str, MCTSNode],
                 orig_smiles: str,
                 clusters: List[Set[int]],
                 atom_cls: List[Set[int]],
                 nei_cls: List[Set[int]],
                 scoring_function: Callable[[List[str]], List[float]],
                 min_atoms:int=8) -> float:
    """
    A Monte Carlo Tree Search rollout from a given :class:`MCTSNode`.

    :param node: The :class:`MCTSNode` from which to begin the rollout.
    :param state_map: A mapping from SMILES to :class:`MCTSNode`.
    :param orig_smiles: The original SMILES of the molecule.
    :param clusters: Clusters of atoms.
    :param atom_cls: Atom indices in the clusters.
    :param nei_cls: Neighboring clusters.
    :param scoring_function: A function for scoring subgraph SMILES using a Chemprop model.
    :return: The score of this MCTS rollout.
    """
    cur_atoms = node.atoms
    if len(cur_atoms) <= min_atoms:
        return node.P

    # Expand if this node has never been visited
    if len(node.children) == 0:
        cur_cls = set([i for i, x in enumerate(clusters) if x <= cur_atoms])
        for i in cur_cls:
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]
            if len(nei_cls[i] & cur_cls) == 1 or len(clusters[i]) == 2 and len(leaf_atoms) == 1:
                new_atoms = cur_atoms - set(leaf_atoms)
                new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)
                if new_smiles in state_map:
                    new_node = state_map[new_smiles]  # merge identical states
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0:
            return node.P  # cannot find leaves

        scores = scoring_function([x.smiles for x in node.children])
        for child, score in zip(node.children, scores):
            child.P = score

    sum_count = sum(c.N for c in node.children)
    selected_node = max(node.children, key=lambda x: x.Q() + x.U(sum_count))
    v = mcts_rollout(selected_node, state_map, orig_smiles, clusters, atom_cls, nei_cls, scoring_function)
    selected_node.W += v
    selected_node.N += 1

    return v


def mcts(smiles: str,
         scoring_function: Callable[[List[str]], List[float]],
         n_rollout: int,
         max_atoms: int,
         prop_delta: float,
         min_atoms:int) -> List[MCTSNode]:
    """
    Runs the Monte Carlo Tree Search algorithm.

    :param smiles: The SMILES of the molecule to perform the search on.
    :param scoring_function: A function for scoring subgraph SMILES using a Chemprop model.
    :param n_rollout: THe number of MCTS rollouts to perform.
    :param max_atoms: The maximum number of atoms allowed in an extracted rationale.
    :param prop_delta: The minimum required property value for a satisfactory rationale.
    :return: A list of rationales each represented by a :class:`MCTSNode`.
    """
            
    mol = Chem.MolFromSmiles(smiles)
    if mol.GetNumAtoms() > 50:
        n_rollout = 1

    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    for i, cls in enumerate(clusters):
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - {i}
        clusters[i] = set(list(cls))
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])

    root = MCTSNode(smiles, set(range(mol.GetNumAtoms())))
    state_map = {smiles: root}
    for _ in range(n_rollout):
        mcts_rollout(root, state_map, smiles, clusters, atom_cls, nei_cls, scoring_function, min_atoms=min_atoms)

    rationales = [node for _, node in state_map.items() if len(node.atoms) <= max_atoms and node.P >= prop_delta]

    return rationales

def mf_scoring_function(smiles:List[str], trainedMF:List) -> List[float]:
    MF_inputs = MolFormerXL_tokenizer(smiles, return_tensors="pt", padding='max_length', max_length=800)
    MF_inputs = MF_inputs.to(device='cuda')
    with torch.no_grad():
        ensemble_y_scores = []
        for model in trainedMF:
            model.to(device='cuda')
            model.eval()
            outputs = model(**MF_inputs)
            ensemble_y_scores.append(outputs[:,1])
        
        ensemble_y_scores = torch.vstack(ensemble_y_scores)
        ensemble_y_scores = 1 / (1 + torch.exp(-ensemble_y_scores))
        agg_y_scores = torch.median(ensemble_y_scores, dim=0).values
    return agg_y_scores.cpu().numpy()
