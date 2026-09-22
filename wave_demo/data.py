# Synthetic data generation for the WAVE demo.
#
# The original WAVE model is trained on real CMap/L1000 gene-expression
# data, which is not available here. To make this app runnable end to
# end without that data, we build a small synthetic dataset: a handful
# of "drug archetypes" each get a fixed random effect vector, and the
# model is trained to recover that vector from (baseline expression,
# drug fingerprint) pairs. This is only meant to demonstrate the WAVE
# architecture and training loop, not to produce real biological
# predictions.

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

N_GENES = 978
FP_BITS = 2048

PRESET_DRUGS = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Kafein (Caffeine)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Parasetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Etanol": "CCO",
}


def morgan_fp(smiles, radius=2, n_bits=FP_BITS):
    """SMILES -> Morgan fingerprint (np.float32 array), or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        generator = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
    except AttributeError:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def random_baseline(n=1, seed=None):
    """Fake baseline (control) gene-expression profiles, non-negative."""
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=2.0, scale=1.5, size=(n, N_GENES)).astype(np.float32)


def build_synthetic_dataset(seed=0, n_cells=300, noise_std=0.3):
    """Build synthetic (baseline, drug_fp, perturbed) training triples.

    Each preset drug gets a fixed random effect vector; the "true"
    perturbed profile for a given baseline is baseline + effect + noise.
    Returns arrays plus the dict of true effect vectors (used to sanity
    check what the model learned).
    """
    rng = np.random.default_rng(seed)
    effect_vectors = {
        name: rng.normal(0, 1.5, size=N_GENES).astype(np.float32)
        for name in PRESET_DRUGS
    }
    fps = {name: morgan_fp(smi) for name, smi in PRESET_DRUGS.items()}

    baselines = random_baseline(n_cells, seed=seed)

    unpert, drug_fp, pert = [], [], []
    for name in PRESET_DRUGS:
        noise = rng.normal(0, noise_std, size=(n_cells, N_GENES)).astype(np.float32)
        target = baselines + effect_vectors[name] + noise
        unpert.append(baselines)
        drug_fp.append(np.tile(fps[name], (n_cells, 1)))
        pert.append(target)

    unpert = np.concatenate(unpert, axis=0)
    drug_fp = np.concatenate(drug_fp, axis=0)
    pert = np.concatenate(pert, axis=0)
    return unpert, drug_fp, pert, effect_vectors
