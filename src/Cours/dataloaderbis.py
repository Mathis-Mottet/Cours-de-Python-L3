from pathlib import Path
import logging
import pickle
from datetime import datetime
from typing import Sequence
import requests
import io

import pandas as pd
import yfinance as yf

from Cours.priceseries import PriceSeries

class DataLoader:
    """
    Charge des données de marché depuis Yahoo Finance avec un système de cache.
    """
    
    def __init__(self, cache_dir: str = ".cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(
            self,
            ticker: str,
            price_col: str,
            dates: tuple[str, str]
            ):
        """
        Génère le chemin du fichier cache pour une requête donnée.
        """
        
        file_name = f"{ticker}_{price_col}_{dates[0]}_{dates[1]}"
        return self.cache_dir / file_name
    
    def _save_to_cache(
            self,
            cache_path: Path,
            ticker: str,
            prices: list[float],
            price_col: str,
            dates: list,
            start: str,
            end: str
            ) -> None:
        """
        Sauvegarde les prix dans un fichier cache avec metadata
        """

        data = {
            "ticker": ticker,
            "start": start,
            "end": end,
            "price_col": price_col,
            "fetched_at": datetime.now().isoformat(),
            "n_prices": len(prices),
            "prices": prices,
            "dates": dates
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    
    def _load_from_cache(
            self,
            ticker: str,
            price_col: str,
            start_date: pd.Timestamp,
            end_date: pd.Timestamp
            ):
        
        if not self.cache_dir.exists():
            print(f"Le dossier cache n'exsite pas")
        for file_path in self.cache_dir.iterdir():
            if not file_path.is_file() or file_path.suffix != '.pkl':
                continue
            name_parts = file_path.stem.split('_')
            if len(name_parts) < 4:
                continue

            cache_ticker = name_parts[0]
            cache_price_col = name_parts[1]
            cache_start_str = pd.Timestamp(name_parts[2])
            cache_end_str = pd.Timestamp(name_parts[3])

            if cache_ticker != ticker or cache_price_col != price_col:
                continue
        pass
        
    
    def _check_date_overlap(
        self,
        cached_start: pd.Timestamp,
        cached_end: pd.Timestamp,
        req_start: pd.Timestamp,
        req_end: pd.Timestamp
    ) -> tuple[str, pd.Timestamp | None, pd.Timestamp | None]:
        """
        Détermine le type de chevauchement entre le cache et la requête
        
        Returns:
            tuple: (status, gap_start, gap_end)
            - status: "exact" | "contains" | "overlap_before" | "overlap_after" | "miss"
            - gap_start: Début de la période manquante (si overlap)
            - gap_end: Fin de la période manquante (si overlap)
        """
        # Cas MISS: Aucune intersection
        if cached_end < req_start or cached_start > req_end:
            return ("miss", None, None)

        # Cas exact: hit parfait du cache
        if cached_start == req_start and cached_end == req_end:
            return ("exact", None, None)

        # Cas CONTAINS: hit du cache qui contient complétement la requête
        if cached_start <= req_start and cached_end >= req_end:
            return ("contains", None, None)

        # Cas OVERLAP_AFTER: cache hit mais la requête débordre à droite
        if cached_start <= req_start and cached_end < req_end:
            gap_start = cached_end + pd.Timedelta(days=1)
            gap_end = req_end
            return ("overlap_after", gap_start, gap_end)

        # Cas OVERLAP_BEFORE: cache hit mais la requête déborde à gauche
        if cached_start > req_start and cached_end >= req_end:
            gap_start = req_start
            gap_end = cached_start - pd.Timedelta(days=1)
            return ("overlap_before", gap_start, gap_end)
        
        return ("miss", None, None)

    

    def fetch_single_loader(
            self,
            ticker: str,
            price_col: str,
            dates: tuple[str, str]
            ) -> PriceSeries:
        """
        Retourne une série de prix
        """
        start_date = pd.Timestamp(dates[0])
        end_date = pd.Timestamp(dates[1])
        ticker_instance = yf.Ticker(ticker)
        df=ticker_instance.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"Dataframe vide pour {ticker}")
            return None
        
        if price_col not in df.columns:
            print(f"{price_col} n'est pas dans le Dataframe de {ticker}")
            raise KeyError (f"{price_col} n'est pas dans le Dataframe de {ticker}")
        
        prices = df.loc[:, price_col]

        if prices.empty:
            print(f"Colonne {price_col} vide pour {ticker}")
            return None
        
        return PriceSeries(values=prices, name=price_col)

if __name__ == "__main__":
    Dataloader = DataLoader()
    result=Dataloader.fetch_single_loader("AAPL", "Close", ("2024-01-01", "2024-06-01"))
    print(result.values)
    pass