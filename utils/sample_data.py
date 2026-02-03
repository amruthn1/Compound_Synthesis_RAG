"""Sample reactions data."""

import pandas as pd
import os


def create_sample_reactions():
    """Create sample reactions.csv file."""
    
    # Sample materials data
    data = {
        'target': [
            'BaTiO3',
            'SrTiO3',
            'LaFeO3',
            'LiCoO2',
            'YBa2Cu3O7',
            'LiFePO4',
            'TiO2',
            'La0.7Sr0.3MnO3',
            'ZrO2',
            'Al2O3'
        ],
        'precursors': [
            'BaCO3, TiO2',
            'SrCO3, TiO2',
            'La2O3, Fe2O3',
            'Li2CO3, Co3O4',
            'Y2O3, BaCO3, CuO',
            'Li2CO3, Fe2O3, NH4H2PO4',
            'Ti metal + O2',
            'La2O3, SrCO3, MnO2',
            'ZrOCl2 calcination',
            'Al(OH)3 calcination'
        ],
        'conditions': [
            '1000-1200°C, 6-12h, air',
            '1200-1400°C, 12h, air',
            '1100-1300°C, 10h, air',
            '800-900°C, 12h, O2',
            '900-950°C, multiple steps, O2',
            '600-800°C, 6-10h, inert',
            '400-700°C, oxidation',
            '1200-1400°C, 12h, air',
            '1000-1200°C, calcination',
            '1200°C, 4h'
        ],
        'category': [
            'perovskite oxide',
            'perovskite oxide',
            'perovskite oxide',
            'battery cathode',
            'superconductor',
            'battery cathode',
            'oxide',
            'perovskite oxide',
            'oxide',
            'oxide'
        ]
    }
    
    df = pd.DataFrame(data)
    return df


def save_sample_reactions(output_path: str):
    """
    Save sample reactions to CSV.
    
    Args:
        output_path: Path to save reactions.csv
    """
    df = create_sample_reactions()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} reactions to {output_path}")


if __name__ == "__main__":
    # Create in data directory
    save_sample_reactions("../data/reactions.csv")
