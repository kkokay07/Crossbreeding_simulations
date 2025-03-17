from src.breed_params import *
from src.trait_params import *
from src.genome import ChickenGenome, GenomeSimulator
from src.breeding import BreedingProgram, BreedingConfig

def main():
    config = BreedingConfig(
        pop_size=100,
        n_generations=10,
        selection_intensity=0.2,
        crossing_rate=0.8,
        mutation_rate=1e-6
    )
    
    program = BreedingProgram(config)
    # Run your simulation...

if __name__ == "__main__":
    main()
