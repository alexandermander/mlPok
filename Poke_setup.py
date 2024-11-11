#No,Mark,Card name,Type,Rarity
#001/142,G,Venusaurex,Grass,Double Rare
#002/142,H,Ledyba,Grass,Common
#003/142,H,Ledian,Grass,Rare
import os
import csv
import re

import requests
from bs4 import BeautifulSoup

class Pokemon:
    def __init__(self, no, mark, card_name, type, rarity, set_name):
        self.set_name = set_name
        self.number = no
        self.mark = mark
        self.name = card_name
        self.type = type
        self.rarity = rarity
        #fuurstnumer and then name
        self.path = f"./Pokemons/{self.set_name}/{self.number.replace('/', '_')}_{self.name}"
        self.urls = []

    def create_folder(self):
        #create a folder for each pokemon
        #the no is 036/142 but use a _ instead of /
        #folder path start ./Pokemon/
        number_format = self.number.replace("/", "_")
        try:
            os.mkdir(self.path)
            print(f"Folder {self.name}_{number_format} created")
        except FileExistsError:
            print("Folder already exists")

    def save_image_path(self, urls):
        self.urls = set(urls)
        print("Saving image path the unique urls are: ", len(self.urls))


    def save_image(self):
        i = 0
        for image in self.urls:
            if image.endswith("500.webp"):
                i += 1
                print(f"Saving image {i}, {image}")
                response = requests.get(image)
                with open(f"{self.path}/{i}.webp", "wb") as file:
                    file.write(response.content)
                    file.close()



def main():
    #input the set name: like "sv_crown" or paladin fatres
    set_name = input("Enter the set name: ")
    try:
        os.mkdir("./Pokemons")
    except FileExistsError:
        print("Folder already exists")
    try:
        os.mkdir(f"./Pokemons/{set_name}")
    except FileExistsError:
        print("Folder already exists")
    current_file = []
    pokemon = []
    with open("./sv_crown.csv", "r") as file:
        current_file = file.readlines()
        file.close()
        csv_reader = csv.reader(current_file)
        #skip the first line
        next(csv_reader)
        for line in csv_reader:
            #print(line)
            pokemon.append(Pokemon(line[0], line[1], line[2], line[3], line[4], set_name))
        file.close()
    print("the first pokemon is: ", pokemon[0].name, pokemon[0].number)
    import findEbay
    #onlt get the first 1 pokemons
#    urls = findEbay.getEbayImageLinks(pokemon[0].name, pokemon[0].number)
#    pokemon[0].save_image_path(urls)
#    print(pokemon[0].urls)
#    print("THen iamges are saved")
#    pokemon[0].save_image()
#    print("Done")
    for p in pokemon:
        #print the pokemin name and number
        print("setting up for ", p.name, p.number)
        p.create_folder()
        print("Getting image links")
        try:
            urls = findEbay.getEbayImageLinks(p.name, p.number)
        except:
            print("Error getting image links")
            continue
        print("Saving image path")
        p.save_image_path(urls)
        p.save_image()
        print("Done" + p.name + p.number)

if __name__ == "__main__":
    main()
