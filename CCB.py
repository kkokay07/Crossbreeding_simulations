# realistic_ccb.py

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
    chromosome: int
    position: int  # Relative position within chromosome
    additive_effect: float
    dominance_effect: float

class MilkQTLs:
    def __init__(self):
        # Real QTL effects from literature
        self.qtls = {
            'DGAT1': QTLInfo(
                chromosome=14,
                position=0.3,  # Relative position
                additive_effect=400,
                dominance_effect=50
            ),
            'GHR': QTLInfo(
                chromosome=20,
                position=0.4,
                additive_effect=300,
                dominance_effect=30
            ),
            'BTA5_1': QTLInfo(
                chromosome=5,
                position=0.2,
                additive_effect=200,
                dominance_effect=20
            ),
            'BTA5_2': QTLInfo(
                chromosome=5,
                position=0.6,
                additive_effect=150,
                dominance_effect=15
            ),
            'BTA6': QTLInfo(
                chromosome=6,
                position=0.5,
                additive_effect=180,
                dominance_effect=18
            ),
            'BTA14_2': QTLInfo(
                chromosome=14,
                position=0.7,
                additive_effect=160,
                dominance_effect=16
            )
        }

class AdaptationQTLs:
    def __init__(self):
        # Heat tolerance and disease resistance QTLs
        self.qtls = {
            'HSF1': QTLInfo(
                chromosome=10,
                position=0.4,
                additive_effect=0.4,
                dominance_effect=0.04
            ),
            'ATP1A1': QTLInfo(
                chromosome=3,
                position=0.3,
                additive_effect=0.3,
                dominance_effect=0.03
            ),
            'SLICK': QTLInfo(
                chromosome=20,
                position=0.6,
                additive_effect=0.5,
                dominance_effect=0.05
            ),
            'BoLA': QTLInfo(
                chromosome=23,
                position=0.5,
                additive_effect=0.45,
                dominance_effect=0.045
            )
        }

@dataclass
class ChromosomeConfig:
    n_chromosomes: int = 30
    base_pairs: List[int] = None
    
    def __post_init__(self):
        if self.base_pairs is None:
            # Generate realistic bovine chromosome sizes (scaled down)
            self.base_pairs = [
                30000, 25000, 24000, 23000, 22000,  # BTA1-5
                20000, 19000, 18000, 17000, 16000,  # BTA6-10
                15000, 14000, 13000, 12000, 11000,  # BTA11-15
                10000, 9000, 8000, 7000, 6000,      # BTA16-20
                5000, 4000, 3000, 2000, 1900,       # BTA21-25
                1800, 1700, 1600, 1500, 1400        # BTA26-30
            ]

class CattleBreedingSimulator:
    def __init__(self, n_base: int = 100):
        self.n_base = n_base
        self.chr_config = ChromosomeConfig()
        self.milk_qtls = MilkQTLs()
        self.adaptation_qtls = AdaptationQTLs()
        
        # Heritabilities from literature
        self.milk_h2 = 0.30
        self.adaptation_h2 = 0.25
        
        # Initialize populations
        print(f"{Fore.GREEN}Initializing HF population...{Style.RESET_ALL}")
        self.hf_pop = self._init_breed('hf')
        print(f"{Fore.GREEN}Initializing Gir population...{Style.RESET_ALL}")
        self.gir_pop = self._init_breed('gir')
    
    def _get_qtl_position(self, chromosome: int, rel_position: float) -> int:
        """Convert relative chromosome position to absolute position"""
        chr_start = sum(self.chr_config.base_pairs[:chromosome-1])
        chr_length = self.chr_config.base_pairs[chromosome-1]
        return chr_start + int(chr_length * rel_position)
    
    def _init_breed(self, breed: str) -> np.ndarray:
        """Initialize breed with appropriate QTL distributions"""
        n_markers = sum(self.chr_config.base_pairs)
        pop = np.zeros((self.n_base, n_markers, 2))
        
        # Add some base variation to all markers
        pop = np.random.binomial(1, 0.1, size=pop.shape)
        
        if breed == 'hf':
            # Set favorable milk QTL alleles
            for qtl in self.milk_qtls.qtls.values():
                pos = self._get_qtl_position(qtl.chromosome, qtl.position)
                # High frequency of favorable alleles in HF
                pop[:, pos] = np.random.binomial(1, 0.9, size=(self.n_base, 2))
        else:  # gir
            # Set favorable adaptation QTL alleles
            for qtl in self.adaptation_qtls.qtls.values():
                pos = self._get_qtl_position(qtl.chromosome, qtl.position)
                # High frequency of favorable alleles in Gir
                pop[:, pos] = np.random.binomial(1, 0.9, size=(self.n_base, 2))
        
        return pop

    def conventional_cross(self, n_offspring: int) -> np.ndarray:
        """Perform conventional crossbreeding"""
        print(f"\n{Fore.GREEN}Performing conventional crosses...{Style.RESET_ALL}")
        offspring = np.zeros((n_offspring, sum(self.chr_config.base_pairs), 2))
        
        for i in tqdm(range(n_offspring), desc="Conventional breeding"):
            hf_parent = self.hf_pop[np.random.randint(self.n_base)]
            gir_parent = self.gir_pop[np.random.randint(self.n_base)]
            
            for haplotype in range(2):
                offspring[i, :, haplotype] = np.where(
                    np.random.random(sum(self.chr_config.base_pairs)) < 0.5,
                    hf_parent[:, np.random.randint(2)],
                    gir_parent[:, np.random.randint(2)]
                )
        
        return offspring

    def ccb_cross(self, n_offspring: int) -> np.ndarray:
        """Perform CCB crossing - targeted chromosomes from HF"""
        print(f"\n{Fore.GREEN}Performing CCB crosses...{Style.RESET_ALL}")
        offspring = np.zeros((n_offspring, sum(self.chr_config.base_pairs), 2))
        
        # Track chromosome starts
        chr_starts = np.cumsum([0] + self.chr_config.base_pairs[:-1])
        
        for i in tqdm(range(n_offspring), desc="CCB breeding"):
            hf_parent = self.hf_pop[np.random.randint(self.n_base)]
            gir_parent = self.gir_pop[np.random.randint(self.n_base)]
            
            # Start with Gir background
            offspring[i] = gir_parent.copy()
            
            # Force milk QTL chromosomes from HF
            target_chrs = {qtl.chromosome for qtl in self.milk_qtls.qtls.values()}
            for chr_num in target_chrs:
                chr_start = chr_starts[chr_num-1]
                chr_end = chr_starts[chr_num-1] + self.chr_config.base_pairs[chr_num-1]
                offspring[i, chr_start:chr_end] = hf_parent[chr_start:chr_end]
        
        return offspring

    def calculate_phenotypes(self, population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate milk and adaptation phenotypes"""
        milk_values = np.zeros(len(population))
        adapt_values = np.zeros(len(population))
        
        # Calculate milk QTL effects
        base_milk = 2800  # Base milk yield
        for qtl in self.milk_qtls.qtls.values():
            pos = self._get_qtl_position(qtl.chromosome, qtl.position)
            genotypes = np.sum(population[:, pos], axis=1)
            
            # Add additive and dominance effects
            milk_values += genotypes * qtl.additive_effect
            milk_values += (genotypes == 1) * qtl.dominance_effect
        
        # Calculate adaptation QTL effects
        for qtl in self.adaptation_qtls.qtls.values():
            pos = self._get_qtl_position(qtl.chromosome, qtl.position)
            genotypes = np.sum(population[:, pos], axis=1)
            
            adapt_values += genotypes * qtl.additive_effect
            adapt_values += (genotypes == 1) * qtl.dominance_effect
        
        # Add environmental variation
        milk_genetic_var = np.var(milk_values)
        adapt_genetic_var = np.var(adapt_values)
        
        milk_env_var = (milk_genetic_var / self.milk_h2) - milk_genetic_var
        adapt_env_var = (adapt_genetic_var / self.adaptation_h2) - adapt_genetic_var
        
        milk_env = np.random.normal(0, np.sqrt(max(0, milk_env_var)), len(population))
        adapt_env = np.random.normal(0, np.sqrt(max(0, adapt_env_var)), len(population))
        
        # Final phenotypes
        milk_phenotypes = base_milk + milk_values + milk_env
        adapt_phenotypes = adapt_values + adapt_env
        
        # Scale adaptation to 0-10
        adapt_phenotypes = (adapt_phenotypes - np.min(adapt_phenotypes)) * 10 / (np.max(adapt_phenotypes) - np.min(adapt_phenotypes))
        
        return milk_phenotypes, adapt_phenotypes

def plot_results(milk_data: pd.DataFrame, adapt_data: pd.DataFrame):
    """Create and save plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Milk yield plot
    sns.boxplot(data=milk_data, x='Breeding', y='Milk', ax=ax1)
    ax1.set_title('Milk Yield Comparison')
    ax1.set_ylabel('Milk Yield (kg/lactation)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Adaptation plot
    sns.boxplot(data=adapt_data, x='Breeding', y='Adaptation', ax=ax2)
    ax2.set_title('Adaptation Score Comparison')
    ax2.set_ylabel('Adaptation Score (0-10)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('breeding_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Set random seed
        np.random.seed(42)
        
        # Initialize simulation
        print(f"\n{Fore.CYAN}Starting CCB Simulation...{Style.RESET_ALL}")
        sim = CattleBreedingSimulator(n_base=100)
        
        # Generate offspring
        conv_offspring = sim.conventional_cross(n_offspring=100)
        ccb_offspring = sim.ccb_cross(n_offspring=100)
        
        # Calculate phenotypes
        print(f"\n{Fore.GREEN}Calculating phenotypes...{Style.RESET_ALL}")
        hf_milk, hf_adapt = sim.calculate_phenotypes(sim.hf_pop)
        gir_milk, gir_adapt = sim.calculate_phenotypes(sim.gir_pop)
        conv_milk, conv_adapt = sim.calculate_phenotypes(conv_offspring)
        ccb_milk, ccb_adapt = sim.calculate_phenotypes(ccb_offspring)
        
        # Prepare results
        milk_data = pd.DataFrame({
            'Breeding': ['HF']*100 + ['Gir']*100 + ['Conventional']*100 + ['CCB']*100,
            'Milk': np.concatenate([hf_milk, gir_milk, conv_milk, ccb_milk])
        })
        
        adapt_data = pd.DataFrame({
            'Breeding': ['HF']*100 + ['Gir']*100 + ['Conventional']*100 + ['CCB']*100,
            'Adaptation': np.concatenate([hf_adapt, gir_adapt, conv_adapt, ccb_adapt])
        })
        
        # Save and plot results
        milk_data.to_csv('milk_results.csv', index=False)
        adapt_data.to_csv('adaptation_results.csv', index=False)
        plot_results(milk_data, adapt_data)
        
        # Print summary
        print(f"\n{Fore.GREEN}Results Summary:{Style.RESET_ALL}")
        print("\nMilk Yield (kg/lactation):")
        print(milk_data.groupby('Breeding')['Milk'].describe().round(2))
        print("\nAdaptation Score (0-10):")
        print(adapt_data.groupby('Breeding')['Adaptation'].describe().round(2))
        
        print(f"\n{Fore.GREEN}Simulation completed!{Style.RESET_ALL}")
        print("Results saved as: breeding_results.png")
        print("Detailed data saved in: milk_results.csv and adaptation_results.csv")
        
    except Exception as e:
        print(f"\n{Fore.RED}Error in simulation: {str(e)}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    main()
