"""
Consomic Cross-Breeding (CCB) Simulation for Beta-Casein in Cattle
Focus: Transfer of BTA6 containing beta-casein QTL cluster from HF to Gir background
Part 1: Core Classes and Setup
"""

import os
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
    """QTL information based on published beta-casein studies"""
    position_mb: float  # Position in Mb on BTA6
    additive_effect: float  # Effect on beta-casein content
    dominance_effect: float  # Dominance effect as proportion of additive
    variance_explained: float  # Proportion of genetic variance explained

class ChromosomeConfig:
    """Configuration for bovine chromosomes"""
    def __init__(self):
        # Chromosome lengths in Mb based on ARS-UCD1.2 assembly
        self.lengths_mb = [
            158.34, 137.06, 121.43, 120.83, 121.19,  # Chr1-5
            119.46,  # Chr6 - contains beta-casein cluster
            112.64, 113.39, 105.71, 104.31,          # Chr7-10
            107.31, 91.16, 84.24, 84.65, 85.30,      # Chr11-15
            81.72, 75.16, 65.13, 63.45, 71.99,       # Chr16-20
            71.60, 61.44, 52.50, 62.71, 42.90,       # Chr21-25
            51.68, 45.41, 46.31, 51.51, 30.00        # Chr26-30
        ]
        self.markers_per_mb = 20  # Marker density
        
    def get_chromosome_markers(self, chr_num: int) -> int:
        """Get number of markers for a chromosome"""
        if chr_num < 1 or chr_num > 30:
            raise ValueError(f"Invalid chromosome number: {chr_num}")
        return int(self.lengths_mb[chr_num - 1] * self.markers_per_mb)
    
    def get_total_markers(self) -> int:
        """Get total number of markers across all chromosomes"""
        return sum(int(length * self.markers_per_mb) for length in self.lengths_mb)

class BetaCaseinQTLs:
    """Beta-casein QTL definitions based on literature"""
    def __init__(self):
        self.qtls = {
            'CSN2_promoter': QTLInfo(
                position_mb=87.14,  # Promoter region
                additive_effect=0.25,
                dominance_effect=0.10,
                variance_explained=0.15
            ),
            'CSN2_coding': QTLInfo(
                position_mb=87.18,  # Main coding region
                additive_effect=0.40,
                dominance_effect=0.15,
                variance_explained=0.30
            ),
            'CSN2_enhancer': QTLInfo(
                position_mb=87.25,  # Enhancer region
                additive_effect=0.20,
                dominance_effect=0.08,
                variance_explained=0.10
            )
        }

class CattleBreedingSimulator:
    """Main simulator class for CCB"""
    
    def __init__(self, n_base: int = 200):
        self.n_base = n_base
        self.chr_config = ChromosomeConfig()
        self.qtls = BetaCaseinQTLs()
        
        # Trait parameters
        self.beta_casein_h2 = 0.45  # Heritability
        self.adaptation_h2 = 0.25
        
        # Base phenotype values
        self.hf_base_casein = 3.2  # %
        self.gir_base_casein = 2.8  # %
        
        print(f"{Fore.GREEN}Initializing populations...{Style.RESET_ALL}")
        self.hf_pop = self._init_breed('hf')
        self.gir_pop = self._init_breed('gir')
    
    def _get_qtl_index(self, position_mb: float) -> int:
        """Convert physical position to marker index"""
        return int(position_mb * self.chr_config.markers_per_mb)
    
    def _init_breed(self, breed: str) -> np.ndarray:
        """Initialize breed-specific genotypes"""
        total_markers = self.chr_config.get_total_markers()
        pop = np.zeros((self.n_base, total_markers, 2))
        
        # Background variation
        pop = np.random.binomial(1, 0.1, size=pop.shape)
        
        # Get BTA6 indices
        bta6_start = sum(self.chr_config.get_chromosome_markers(i) for i in range(1, 6))
        bta6_length = self.chr_config.get_chromosome_markers(6)
        
        # Set QTL allele frequencies based on breed
        freq = 0.9 if breed == 'hf' else 0.2  # HF has high frequency of favorable alleles
        
        for qtl in self.qtls.qtls.values():
            pos = bta6_start + self._get_qtl_index(qtl.position_mb)
            if pos >= bta6_start + bta6_length:
                raise ValueError(f"QTL position {qtl.position_mb}Mb exceeds chromosome length")
            pop[:, pos] = np.random.binomial(1, freq, size=(self.n_base, 2))
        
        return pop
    
    def _calculate_beta_casein(self, genotypes: np.ndarray) -> float:
        """Calculate beta-casein phenotype"""
        base_value = 2.8  # Base beta-casein content
        
        # Get BTA6 start index
        bta6_start = sum(self.chr_config.get_chromosome_markers(i) for i in range(1, 6))
        
        # Calculate QTL effects
        for qtl in self.qtls.qtls.values():
            pos = bta6_start + self._get_qtl_index(qtl.position_mb)
            n_favorable = np.sum(genotypes[pos])
            
            # Add additive effect
            base_value += n_favorable * qtl.additive_effect
            
            # Add dominance deviation if heterozygous
            if n_favorable == 1:
                base_value += qtl.additive_effect * qtl.dominance_effect
        
        # Extra effect for complete favorable haplotype
        all_favorable = True
        for qtl in self.qtls.qtls.values():
            pos = bta6_start + self._get_qtl_index(qtl.position_mb)
            if np.sum(genotypes[pos]) < 2:
                all_favorable = False
                break
        
        if all_favorable:
            base_value *= 1.1  # 10% bonus for complete favorable haplotype
        
        return base_value
    
    def conventional_cross(self, n_offspring: int) -> np.ndarray:
        """Perform conventional crossbreeding"""
        print(f"\n{Fore.GREEN}Performing conventional crosses...{Style.RESET_ALL}")
        
        total_markers = self.chr_config.get_total_markers()
        offspring = np.zeros((n_offspring, total_markers, 2))
        
        for i in tqdm(range(n_offspring)):
            # Select parents
            hf_parent = self.hf_pop[np.random.randint(self.n_base)]
            gir_parent = self.gir_pop[np.random.randint(self.n_base)]
            
            # Random inheritance per chromosome
            marker_start = 0
            for chr_num in range(1, 31):
                n_markers = self.chr_config.get_chromosome_markers(chr_num)
                marker_end = marker_start + n_markers
                
                # Random segregation
                offspring[i, marker_start:marker_end, 0] = hf_parent[marker_start:marker_end, np.random.randint(2)]
                offspring[i, marker_start:marker_end, 1] = gir_parent[marker_start:marker_end, np.random.randint(2)]
                
                marker_start = marker_end
        
        return offspring
    
    def ccb_cross(self, n_offspring: int) -> np.ndarray:
        """Perform CCB crossing - transfer BTA6 from HF"""
        print(f"\n{Fore.GREEN}Performing CCB crosses...{Style.RESET_ALL}")
        
        total_markers = self.chr_config.get_total_markers()
        offspring = np.zeros((n_offspring, total_markers, 2))
        
        # Get BTA6 indices
        bta6_start = sum(self.chr_config.get_chromosome_markers(i) for i in range(1, 6))
        bta6_end = bta6_start + self.chr_config.get_chromosome_markers(6)
        
        for i in tqdm(range(n_offspring)):
            # Select best HF parent for beta-casein
            best_hf_parent = None
            best_casein = -float('inf')
            
            for _ in range(5):  # Sample 5 potential parents
                candidate = self.hf_pop[np.random.randint(self.n_base)]
                casein_value = self._calculate_beta_casein(candidate)
                if casein_value > best_casein:
                    best_casein = casein_value
                    best_hf_parent = candidate
            
            # Select best Gir parent for adaptation
            best_gir_parent = None
            best_adapt = -float('inf')
            
            for _ in range(5):
                candidate = self.gir_pop[np.random.randint(self.n_base)]
                chr6_markers = candidate[bta6_start:bta6_end]
                adapt_score = np.mean(chr6_markers)  # Simple adaptation score
                if adapt_score > best_adapt:
                    best_adapt = adapt_score
                    best_gir_parent = candidate
            
            # Create offspring with Gir background
            offspring[i] = best_gir_parent.copy()
            
            # Replace BTA6 with HF version
            offspring[i, bta6_start:bta6_end] = best_hf_parent[bta6_start:bta6_end]
        
        return offspring
    
    def calculate_phenotypes(self, population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate beta-casein and adaptation phenotypes with improved adaptation scoring"""
        n_animals = len(population)
        
        # Calculate beta-casein values
        casein_values = np.array([self._calculate_beta_casein(animal) for animal in population])
        
        # Calculate improved adaptation scores
        bta6_start = sum(self.chr_config.get_chromosome_markers(i) for i in range(1, 6))
        bta6_end = bta6_start + self.chr_config.get_chromosome_markers(6)
        
        adapt_values = np.zeros(n_animals)
        for i in range(n_animals):
            # Heat tolerance (higher for Gir-like patterns)
            non_bta6 = np.concatenate([
                population[i, :bta6_start],
                population[i, bta6_end:]
            ])
            # Invert the score since Gir has lower frequency of HF alleles
            heat_tolerance = (1 - np.mean(non_bta6)) * 4  # Scale to 0-4
            
            # Disease resistance (specific to tropical diseases)
            disease_markers = population[i, bta6_start:bta6_end:10]
            # Invert score to favor Gir patterns
            disease_resistance = (1 - np.mean(disease_markers)) * 3  # Scale to 0-3
            
            # Metabolic efficiency in tropical conditions
            efficiency_markers = population[i, ::20]
            # Invert score to favor Gir patterns
            metabolic_efficiency = (1 - np.mean(efficiency_markers)) * 3  # Scale to 0-3
            
            # Combine scores with weights
            adapt_values[i] = (0.4 * heat_tolerance + 
                              0.3 * disease_resistance + 
                              0.3 * metabolic_efficiency)
        
        # Add environmental variation
        casein_genetic_var = np.var(casein_values)
        casein_env_var = (casein_genetic_var / self.beta_casein_h2) - casein_genetic_var
        casein_env = np.random.normal(0, np.sqrt(casein_env_var), n_animals)
        
        adapt_genetic_var = np.var(adapt_values)
        adapt_env_var = (adapt_genetic_var / self.adaptation_h2) - adapt_genetic_var
        adapt_env = np.random.normal(0, np.sqrt(adapt_env_var), n_animals)
        
        return casein_values + casein_env, adapt_values + adapt_env

def plot_results(casein_data: pd.DataFrame, adapt_data: pd.DataFrame):
    """Create publication-quality plots with consistent data handling"""
    try:
        # Create figure with appropriate size
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Style settings
        plt.style.use('default')
        colors = ["#2ecc71", "#e74c3c", "#3498db", "#f1c40f"]
        
        # Ensure consistent order of breeding strategies
        breed_order = ['HF', 'Gir', 'Conventional', 'CCB']
        
        # Create plot data in consistent order
        casein_plot_data = [casein_data[casein_data['Breeding'] == breed]['Beta_Casein'].values 
                           for breed in breed_order]
        
        # Beta-casein plot
        bp1 = ax1.boxplot(casein_plot_data,
                         labels=breed_order,
                         patch_artist=True,
                         medianprops=dict(color="black"),
                         boxprops=dict(color="black"))
                   
        # Color the boxes
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            
        ax1.set_title('Beta-Casein Content', fontsize=12, pad=10)
        ax1.set_ylabel('Beta-Casein Content (%)', fontsize=10)
        ax1.set_xlabel('Breeding Strategy', fontsize=10)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Adaptation plot
        adapt_plot_data = [adapt_data[adapt_data['Breeding'] == breed]['Adaptation'].values 
                          for breed in breed_order]
        
        bp2 = ax2.boxplot(adapt_plot_data,
                         labels=breed_order,
                         patch_artist=True,
                         medianprops=dict(color="black"),
                         boxprops=dict(color="black"))
                   
        # Color the boxes
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            
        ax2.set_title('Tropical Adaptation Score', fontsize=12, pad=10)
        ax2.set_ylabel('Adaptation Score (0-10)', fontsize=10)
        ax2.set_xlabel('Breeding Strategy', fontsize=10)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs('simulation_results', exist_ok=True)
        
        # Save plot
        plt.savefig('simulation_results/beta_casein_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save data files with proper paths
        casein_data.to_csv('simulation_results/beta_casein_data.csv', index=False)
        adapt_data.to_csv('simulation_results/adaptation_data.csv', index=False)
        
        # Save summary statistics
        summary_stats = pd.DataFrame({
            'Beta_Casein_Mean': casein_data.groupby('Breeding')['Beta_Casein'].mean(),
            'Beta_Casein_SD': casein_data.groupby('Breeding')['Beta_Casein'].std(),
            'Adaptation_Mean': adapt_data.groupby('Breeding')['Adaptation'].mean(),
            'Adaptation_SD': adapt_data.groupby('Breeding')['Adaptation'].std()
        }).round(3)
        summary_stats.to_csv('simulation_results/summary_stats.csv')
        
        print(f"\nResults saved in 'simulation_results' directory:")
        print("- beta_casein_results.png")
        print("- beta_casein_data.csv")
        print("- adaptation_data.csv")
        print("- summary_stats.csv")
        
    except Exception as e:
        print(f"Error in plot_results: {str(e)}")
        plt.close()
        raise
def main():
    """Run simulation with improved error handling and data validation"""
    try:
        # Create output directory
        output_dir = "simulation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Initialize simulation
        print(f"\n{Fore.CYAN}Starting Beta-Casein CCB Simulation...{Style.RESET_ALL}")
        sim = CattleBreedingSimulator(n_base=200)
        
        # Validate initial populations
        if sim.hf_pop is None or sim.gir_pop is None:
            raise ValueError("Failed to initialize breeding populations")
            
        if sim.hf_pop.shape != (200, sim.chr_config.get_total_markers(), 2):
            raise ValueError(f"Invalid HF population shape: {sim.hf_pop.shape}")
        if sim.gir_pop.shape != (200, sim.chr_config.get_total_markers(), 2):
            raise ValueError(f"Invalid Gir population shape: {sim.gir_pop.shape}")
        
        # Generate offspring with progress tracking
        n_offspring = 200
        print("\nGenerating offspring populations...")
        
        print("\nPerforming conventional crosses...")
        conv_offspring = sim.conventional_cross(n_offspring)
        if conv_offspring is None or conv_offspring.shape != (n_offspring, sim.chr_config.get_total_markers(), 2):
            raise ValueError(f"Invalid conventional offspring population shape: {conv_offspring.shape if conv_offspring is not None else None}")
            
        print("\nPerforming CCB crosses...")
        ccb_offspring = sim.ccb_cross(n_offspring)
        if ccb_offspring is None or ccb_offspring.shape != (n_offspring, sim.chr_config.get_total_markers(), 2):
            raise ValueError(f"Invalid CCB offspring population shape: {ccb_offspring.shape if ccb_offspring is not None else None}")
        
        # Calculate phenotypes with validation
        print(f"\n{Fore.GREEN}Calculating phenotypes...{Style.RESET_ALL}")
        
        # Calculate and validate phenotypes for each population
        populations = {
            'HF': sim.hf_pop,
            'Gir': sim.gir_pop,
            'Conventional': conv_offspring,
            'CCB': ccb_offspring
        }
        
        casein_results = {}
        adapt_results = {}
        
        for pop_name, pop in populations.items():
            print(f"\nProcessing {pop_name} population...")
            casein_vals, adapt_vals = sim.calculate_phenotypes(pop)
            
            # Validate results
            if len(casein_vals) != len(pop) or len(adapt_vals) != len(pop):
                raise ValueError(f"Phenotype calculation failed for {pop_name}")
                
            casein_results[pop_name] = casein_vals
            adapt_results[pop_name] = adapt_vals
        
        # Prepare data for visualization
        print("\nPreparing data for visualization...")
        casein_data_list = []
        adapt_data_list = []
        
        for pop_name in populations.keys():
            casein_data_list.append(pd.DataFrame({
                'Breeding': [pop_name] * len(casein_results[pop_name]),
                'Beta_Casein': casein_results[pop_name]
            }))
            
            adapt_data_list.append(pd.DataFrame({
                'Breeding': [pop_name] * len(adapt_results[pop_name]),
                'Adaptation': adapt_results[pop_name]
            }))
        
        casein_data = pd.concat(casein_data_list, ignore_index=True)
        adapt_data = pd.concat(adapt_data_list, ignore_index=True)
        
        # Calculate and display summary statistics
        print(f"\n{Fore.GREEN}Summary Statistics:{Style.RESET_ALL}")
        
        print("\nBeta-Casein Content (%):")
        casein_summary = casein_data.groupby('Breeding')['Beta_Casein'].describe()
        print(casein_summary.round(3))
        
        print("\nAdaptation Score (0-10):")
        adapt_summary = adapt_data.groupby('Breeding')['Adaptation'].describe()
        print(adapt_summary.round(3))
        
        # Save detailed results
        print("\nSaving results...")
        casein_data.to_csv(os.path.join(output_dir, 'beta_casein_results.csv'), index=False)
        adapt_data.to_csv(os.path.join(output_dir, 'adaptation_results.csv'), index=False)
        
        # Create visualization
        print(f"\n{Fore.GREEN}Generating visualization...{Style.RESET_ALL}")
        plot_results(casein_data, adapt_data)
        
        # Save summary statistics
        summary_stats = pd.DataFrame({
            'Beta_Casein_Mean': casein_data.groupby('Breeding')['Beta_Casein'].mean(),
            'Beta_Casein_SD': casein_data.groupby('Breeding')['Beta_Casein'].std(),
            'Adaptation_Mean': adapt_data.groupby('Breeding')['Adaptation'].mean(),
            'Adaptation_SD': adapt_data.groupby('Breeding')['Adaptation'].std()
        }).round(3)
        
        summary_stats.to_csv(os.path.join(output_dir, 'summary_stats.csv'))
        
        print(f"\n{Fore.GREEN}Simulation completed successfully!{Style.RESET_ALL}")
        print(f"\nResults saved in: {output_dir}")
        print("\nOutput files:")
        print("- beta_casein_results.png: Visualization of results")
        print("- beta_casein_results.csv: Detailed beta-casein data")
        print("- adaptation_results.csv: Detailed adaptation scores")
        print("- summary_stats.csv: Statistical summary")
        
    except Exception as e:
        print(f"\n{Fore.RED}Error in simulation: {str(e)}{Style.RESET_ALL}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        # Clean up matplotlib resources
        plt.close('all')

if __name__ == "__main__":
    main()
