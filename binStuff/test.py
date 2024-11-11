

#createa  clase called: MlPokemon
import os

class MlPokemon:
    def __init__(self,  path):
        self.path = path
        self.images = os.listdir(self.path)
        self.traing_images = []

    def getprint(self):
        print(self.images) 


if __name__ == "__main__":
    #in the 
    path = "./Pokemons/Stellar_Crown/"
    pokepaths = []
    for paths in os.listdir(path):
        print(paths)
        pokepaths.append(path + paths)
    for pokepath in pokepaths:
        print(pokepath)
        pokemon = MlPokemon(pokepath)
        pokemon.getprint()

#Pokemonsi


