# Fichier: pyvest/core/universe.py

from Cours.asset import Asset
from typing import Iterator


class Universe:
    """
    Collection d'actifs représentant un univers d'investissement.
    
    Pattern de conception : AGRÉGATION
    ──────────────────────────────────
    Universe CONTIENT des Asset, mais les Asset peuvent exister 
    indépendamment de l'Universe.
    
    La classe implémente le protocole d'itération (__iter__) et
    de conteneur (__contains__, __len__) pour une utilisation
    pythonique.
    """
    
    def __init__(self, assets: list[Asset] | None = None) -> None:
        self._assets: dict[str, Asset] = {}
        if assets:
            for asset in assets:
                self.add(asset)
    
    def add(self, asset: Asset) -> None:
        """Ajoute un actif à l'univers."""
        # Votre code ici
        pass
    
    def get(self, ticker: str) -> Asset | None:
        """Récupère un actif par son ticker."""
        # Votre code ici
        pass
    
    def remove(self, ticker: str) -> Asset | None:
        """Retire un actif de l'univers."""
        # Votre code ici
        pass
    
    def __len__(self) -> int:
        # Votre code ici
        pass
    
    def __iter__(self) -> Iterator[Asset]:
        # Votre code ici
        pass
    
    def __contains__(self, ticker: str) -> bool:
        # Votre code ici
        pass
    
    @property
    def tickers(self) -> list[str]:
        # Votre code ici
        pass
    
    def filter_by_sector(self, sector: str) -> list[Asset]:
        """Filtre les actifs par secteur."""
        # Votre code ici
        pass