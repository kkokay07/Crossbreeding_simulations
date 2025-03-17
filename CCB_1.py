# improved_ccb.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple, Dict
from tqdm import tqdm
from colorama import Fore, Style, init
init()

@dataclass
class QTLInfo:
    """QTL information based on published cattle studies"""
    chromosome: int
    position: int  # Relative position within chromosome
    additive_effect: float  # Additive effect in kg for milk, or units for adaptation
    dominance_effect: float  # Dominance effect as proportion of additive effect
    variance_explained: float  # Proportion of genetic variance explained

class MajorQTLs:
    def __init__(self):
        """
        Initialize major QTLs with realistic effects from literature.
        References:
        - DGAT1: Grisart et al. 2004
        - GHR: Blott et al. 2003
        - ABCG2: Cohen-Zinder et al. 2005
        - SCD1: Schennink et al. 2008
        """
        # Milk production QTLs
        self.milk_qtls = {
            'DGAT1': QTLInfo(
                chromosome=14,
                position=0.25,  # 25% along chromosome
                additive_effect=225.0,  # kg/lactation
                dominance_effect=0.12,  # Partial dominance
                variance_explained=0.15  # Explains 15% of genetic variance
            ),
            'GHR': QTLInfo(
                chromosome=20,
                position=0.35,
                additive_effect=180.0,
                dominance_effect=0.08,
                variance_explained=0.10
            ),
            'ABCG2': QTLInfo(
                chromosome=6,
                position=0.45,
                additive_effect=150.0,
                dominance_effect=0.06,
                variance_explained=0.08
            ),
            'SCD1': QTLInfo(
                chromosome=26,
                position=0.30,
                additive_effect=120.0,
                dominance_effect=0.05,
                variance_explained=0.06
            )
        }
        
        # Heat tolerance & adaptation QTLs
        self.adaptation_qtls = {
            'HSF1': QTLInfo(
                chromosome=10,
                position=0.40,
                additive_effect=0.45,  # Standardized effect
                dominance_effect=0.15,
                variance_explained=0.12
            ),
            'ATP1A1': QTLInfo(
                chromosome=3,
                position=0.35,
                additive_effect=0.35,
                dominance_effect=0.10,
                variance_explained=0.09
            ),
            'SLICK': QTLInfo(
                chromosome=20,
                position=0.55,
                additive_effect=0.50,
                dominance_effect=0.20,
                variance_explained=0.15
            ),
            'BoLA': QTLInfo(
                chromosome=23,
                position=0.45,
                additive_effect=0.40,
                dominance_effect=0.12,
                variance_explained=0.10
            )
        }

@dataclass
class ChromosomeConfig:
    """Realistic bovine chromosome configuration"""
    n_chromosomes: int = 30
    morgan_lengths: List[float] = None
    
    def __post_init__(self):
        if self.morgan_lengths is None:
            # Morgan lengths based on cattle genetic map (Arias et al. 2009)
            self.morgan_lengths = [
                1.59, 1.36, 1.41, 1.27, 1.25,  # BTA1-5
                1.33, 1.12, 1.17, 1.07, 1.04,  # BTA6-10
                1.21, 0.99, 0.98, 0.86, 0.91,  # BTA11-15
                0.87, 0.84, 0.81, 0.75, 0.72,  # BTA16-20
                0.71, 0.69, 0.63, 0.65, 0.61,  # BTA21-25
                0.57, 0.55, 0.53, 0.51, 0.47   # BTA26-30
            ]

class CattleBreedingSimulator:
    def __init__(self, n_base: int = 200):
        """
        Initialize breeding simulation with realistic parameters.
        
        Args:
            n_base: Number of base population animals
        """
        self.n_base = n_base
        self.chr_config = ChromosomeConfig()
        self.qtls = MajorQTLs()
        
        # Heritabilities from large-scale cattle studies
        self.milk_h2 = 0.35  # Meta-analysis value
        self.adaptation_h2 = 0.22  # Lower h2 for fitness traits
        
        # Base genetic variances (scaled to kg^2 for milk)
        self.base_milk_var = 240000  # ~490kg SD
        self.base_adapt_var = 1.0  # Standardized scale
        
        # Initialize populations
        print(f"{Fore.GREEN}Initializing populations...{Style.RESET_ALL}")
        self.hf_pop = self._init_breed('hf')
        self.gir_pop = self._init_breed('gir')
        
    def _get_qtl_index(self, chromosome: int, rel_position: float) -> int:
        """Convert chromosome and relative position to QTL index"""
        chr_starts = np.cumsum([0] + self.chr_config.morgan_lengths[:-1])
        return int(chr_starts[chromosome-1] + rel_position * self.chr_config.morgan_lengths[chromosome-1] * 100)
    
    def _init_breed(self, breed: str) -> np.ndarray:
        """Initialize breed with realistic allele frequencies"""
        n_loci = int(sum(self.chr_config.morgan_lengths) * 100)  # 100 markers per Morgan
        pop = np.zeros((self.n_base, n_loci, 2))
        
        # Base population polymorphism
        pop = np.random.binomial(1, 0.3, size=pop.shape)
        
        if breed == 'hf':
            # High frequency of favorable milk QTL alleles in HF
            for qtl in self.qtls.milk_qtls.values():
                pos = self._get_qtl_index(qtl.chromosome, qtl.position)
                freq = 0.85 + 0.1 * np.random.random()  # 0.85-0.95 frequency
                pop[:, pos] = np.random.binomial(1, freq, size=(self.n_base, 2))
                
            # Lower frequency of adaptation alleles
            for qtl in self.qtls.adaptation_qtls.values():
                pos = self._get_qtl_index(qtl.chromosome, qtl.position)
                freq = 0.15 + 0.1 * np.random.random()  # 0.15-0.25 frequency
                pop[:, pos] = np.random.binomial(1, freq, size=(self.n_base, 2))
        
        else:  # gir
            # High frequency of adaptation QTL alleles
            for qtl in self.qtls.adaptation_qtls.values():
                pos = self._get_qtl_index(qtl.chromosome, qtl.position)
                freq = 0.80 + 0.15 * np.random.random()  # 0.80-0.95 frequency
                pop[:, pos] = np.random.binomial(1, freq, size=(self.n_base, 2))
                
            # Lower frequency of milk production alleles
            for qtl in self.qtls.milk_qtls.values():
                pos = self._get_qtl_index(qtl.chromosome, qtl.position)
                freq = 0.10 + 0.15 * np.random.random()  # 0.10-0.25 frequency
                pop[:, pos] = np.random.binomial(1, freq, size=(self.n_base, 2))
        
        return pop

    def _calculate_milk_value(self, genotypes: np.ndarray) -> float:
        """Calculate milk production value from QTL genotypes with proper epistatic effects"""
        # Higher base value for improved breeds
        value = 5000  # Base milk yield for modern dairy cattle
        
        # Track complete chromosome effects
        chr_effects = {}
        
        for qtl_name, qtl in self.qtls.milk_qtls.items():
            pos = self._get_qtl_index(qtl.chromosome, qtl.position)
            n_favorable = np.sum(genotypes[pos])
            
            # Add additive effect with enhanced effect when chromosome is complete
            if qtl.chromosome not in chr_effects:
                chr_effects[qtl.chromosome] = []
            chr_effects[qtl.chromosome].append(n_favorable)
            
            # Basic additive effect
            value += n_favorable * qtl.additive_effect
            
            # Add dominance deviation if heterozygous
            if n_favorable == 1:
                value += qtl.additive_effect * qtl.dominance_effect
        
        # Add synergistic effects for complete chromosomes
        for chr_num, effects in chr_effects.items():
            if all(e == 2 for e in effects):  # All favorable alleles present
                value *= 1.15  # 15% boost for complete chromosome networks
                
        return value

    def _calculate_adaptation_value(self, genotypes: np.ndarray) -> float:
        """Calculate adaptation value from QTL genotypes"""
        value = 0  # Base adaptation score
        
        for qtl in self.qtls.adaptation_qtls.values():
            pos = self._get_qtl_index(qtl.chromosome, qtl.position)
            n_favorable = np.sum(genotypes[pos])
            
            # Add additive effect
            value += n_favorable * qtl.additive_effect
            
            # Add dominance deviation if heterozygous
            if n_favorable == 1:
                value += qtl.additive_effect * qtl.dominance_effect
                
        return value

    def conventional_cross(self, n_offspring: int) -> np.ndarray:
        """Perform conventional crossbreeding with random inheritance"""
        print(f"\n{Fore.GREEN}Performing conventional crosses...{Style.RESET_ALL}")
        n_loci = int(sum(self.chr_config.morgan_lengths) * 100)
        offspring = np.zeros((n_offspring, n_loci, 2))
        
        for i in tqdm(range(n_offspring)):
            # Select random parents
            hf_parent = self.hf_pop[np.random.randint(self.n_base)]
            gir_parent = self.gir_pop[np.random.randint(self.n_base)]
            
            # Inherit one chromosome copy from each parent
            for chrom in range(self.chr_config.n_chromosomes):
                start_idx = self._get_qtl_index(chrom+1, 0)
                end_idx = self._get_qtl_index(chrom+2, 0) if chrom < 29 else n_loci
                
                # Random segregation
                offspring[i, start_idx:end_idx, 0] = hf_parent[start_idx:end_idx, np.random.randint(2)]
                offspring[i, start_idx:end_idx, 1] = gir_parent[start_idx:end_idx, np.random.randint(2)]
        
        return offspring

    def ccb_cross(self, n_offspring: int) -> np.ndarray:
        """Perform CCB crossing with complete preservation of milk QTL chromosomes"""
        print(f"\n{Fore.GREEN}Performing CCB crosses...{Style.RESET_ALL}")
        n_loci = int(sum(self.chr_config.morgan_lengths) * 100)
        offspring = np.zeros((n_offspring, n_loci, 2))
        
        # Identify chromosomes with major milk QTLs
        milk_chr = {qtl.chromosome for qtl in self.qtls.milk_qtls.values()}
        milk_chr_groups = {}  # Group QTLs by chromosome
        
        # Group QTLs by chromosome for synergistic effects
        for qtl_name, qtl in self.qtls.milk_qtls.items():
            if qtl.chromosome not in milk_chr_groups:
                milk_chr_groups[qtl.chromosome] = []
            milk_chr_groups[qtl.chromosome].append(qtl)
        
        for i in tqdm(range(n_offspring)):
            # Select best HF parent for milk QTLs
            best_hf_score = -float('inf')
            best_hf_parent = None
            
            # Sample 5 potential HF parents
            for _ in range(5):
                candidate = self.hf_pop[np.random.randint(self.n_base)]
                milk_score = self._calculate_milk_value(candidate)
                if milk_score > best_hf_score:
                    best_hf_score = milk_score
                    best_hf_parent = candidate
            
            # Select best Gir parent for adaptation
            best_gir_score = -float('inf')
            best_gir_parent = None
            
            # Sample 5 potential Gir parents
            for _ in range(5):
                candidate = self.gir_pop[np.random.randint(self.n_base)]
                adapt_score = self._calculate_adaptation_value(candidate)
                if adapt_score > best_gir_score:
                    best_gir_score = adapt_score
                    best_gir_parent = candidate
            
            # Start with Gir background
            offspring[i] = best_gir_parent.copy()
            
            # Replace complete milk QTL chromosomes with HF
            for chrom in milk_chr:
                start_idx = self._get_qtl_index(chrom, 0)
                end_idx = self._get_qtl_index(chrom+1, 0) if chrom < 30 else n_loci
                # Transfer both copies to maintain complete networks
                offspring[i, start_idx:end_idx] = best_hf_parent[start_idx:end_idx]
        
        return offspring

    def calculate_phenotypes(self, population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate phenotypes with realistic environmental variation"""
        n_animals = len(population)
        milk_values = np.zeros(n_animals)
        adapt_values = np.zeros(n_animals)
        
        # Calculate genetic values
        for i in range(n_animals):
            milk_values[i] = self._calculate_milk_value(population[i])
            adapt_values[i] = self._calculate_adaptation_value(population[i])
        
        # Add environmental variation
        milk_genetic_var = np.var(milk_values)
        milk_env_var = (milk_genetic_var / self.milk_h2) - milk_genetic_var
        milk_env = np.random.normal(0, np.sqrt(milk_env_var), n_animals)
        
        adapt_genetic_var = np.var(adapt_values)
        adapt_env_var = (adapt_genetic_var / self.adaptation_h2) - adapt_genetic_var
        adapt_env = np.random.normal(0, np.sqrt(adapt_env_var), n_animals)
        
        # Final phenotypes
        milk_phenotypes = milk_values + milk_env
        adapt_phenotypes = adapt_values + adapt_env
        
        # Scale adaptation to 0-10 for interpretability
        adapt_phenotypes = (adapt_phenotypes - np.min(adapt_phenotypes)) * 10 / (np.max(adapt_phenotypes) - np.min(adapt_phenotypes))
        
        return milk_phenotypes, adapt_phenotypes

def plot_results(milk_data: pd.DataFrame, adapt_data: pd.DataFrame):
    """Create publication-quality plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Custom style
    sns.set_style("whitegrid")
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f1c40f"]
    
    # Milk yield plot
    sns.boxplot(data=milk_data, x='Breeding', y='Milk', ax=ax1, palette=colors)
    ax1.set_title('Milk Yield Comparison', fontsize=12, pad=15)
    ax1.set_ylabel('Milk Yield (kg/lactation)', fontsize=10)
    ax1.set_xlabel('Breeding Strategy', fontsize=10)
    
    # Add statistical annotations
    milk_means = milk_data.groupby('Breeding')['Milk'].mean()
    y_pos = milk_data['Milk'].max() + 100
    for i, breed in enumerate(milk_means.index):
        ax1.text(i, y_pos, f'μ={milk_means[breed]:.0f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Adaptation plot
    sns.boxplot(data=adapt_data, x='Breeding', y='Adaptation', ax=ax2, palette=colors)
    ax2.set_title('Heat Tolerance & Disease Resistance', fontsize=12, pad=15)
    ax2.set_ylabel('Adaptation Score (0-10)', fontsize=10)
    ax2.set_xlabel('Breeding Strategy', fontsize=10)
    
    # Add statistical annotations
    adapt_means = adapt_data.groupby('Breeding')['Adaptation'].mean()
    y_pos = adapt_data['Adaptation'].max() + 0.5
    for i, breed in enumerate(adapt_means.index):
        ax2.text(i, y_pos, f'μ={adapt_means[breed]:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # Overall styling
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('breeding_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_summary_stats(milk_data: pd.DataFrame, adapt_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for both traits"""
    summary = pd.DataFrame()
    
    # Milk yield stats
    milk_stats = milk_data.groupby('Breeding')['Milk'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(1)
    milk_stats.columns = [f'Milk_{col}' for col in milk_stats.columns]
    
    # Adaptation stats
    adapt_stats = adapt_data.groupby('Breeding')['Adaptation'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    adapt_stats.columns = [f'Adapt_{col}' for col in adapt_stats.columns]
    
    # Combine stats
    summary = pd.concat([milk_stats, adapt_stats], axis=1)
    return summary

def main():
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Initialize simulation
        print(f"\n{Fore.CYAN}Initializing CCB Simulation...{Style.RESET_ALL}")
        print("Using realistic QTL effects and genetic architecture")
        sim = CattleBreedingSimulator(n_base=200)
        
        # Generate offspring
        n_offspring = 200  # Larger sample size for better statistics
        conv_offspring = sim.conventional_cross(n_offspring)
        ccb_offspring = sim.ccb_cross(n_offspring)
        
        # Calculate phenotypes
        print(f"\n{Fore.GREEN}Calculating phenotypic values...{Style.RESET_ALL}")
        hf_milk, hf_adapt = sim.calculate_phenotypes(sim.hf_pop)
        gir_milk, gir_adapt = sim.calculate_phenotypes(sim.gir_pop)
        conv_milk, conv_adapt = sim.calculate_phenotypes(conv_offspring)
        ccb_milk, ccb_adapt = sim.calculate_phenotypes(ccb_offspring)
        
        # Prepare results dataframes
        milk_data = pd.DataFrame({
            'Breeding': ['HF']*len(hf_milk) + ['Gir']*len(gir_milk) + 
                       ['Conventional']*len(conv_milk) + ['CCB']*len(ccb_milk),
            'Milk': np.concatenate([hf_milk, gir_milk, conv_milk, ccb_milk])
        })
        
        adapt_data = pd.DataFrame({
            'Breeding': ['HF']*len(hf_adapt) + ['Gir']*len(gir_adapt) + 
                       ['Conventional']*len(conv_adapt) + ['CCB']*len(ccb_adapt),
            'Adaptation': np.concatenate([hf_adapt, gir_adapt, conv_adapt, ccb_adapt])
        })
        
        # Calculate and display summary statistics
        print(f"\n{Fore.GREEN}Summary Statistics:{Style.RESET_ALL}")
        summary_stats = calculate_summary_stats(milk_data, adapt_data)
        print("\n", summary_stats)
        
        # Save results
        milk_data.to_csv('milk_results.csv', index=False)
        adapt_data.to_csv('adaptation_results.csv', index=False)
        summary_stats.to_csv('summary_stats.csv')
        
        # Create plots
        print(f"\n{Fore.GREEN}Generating visualization...{Style.RESET_ALL}")
        plot_results(milk_data, adapt_data)
        
        print(f"\n{Fore.GREEN}Simulation completed successfully!{Style.RESET_ALL}")
        print("Output files:")
        print("- breeding_results.png: Visualization of results")
        print("- milk_results.csv: Detailed milk yield data")
        print("- adaptation_results.csv: Detailed adaptation score data")
        print("- summary_stats.csv: Statistical summary")
        
    except Exception as e:
        print(f"\n{Fore.RED}Error in simulation: {str(e)}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    main()
