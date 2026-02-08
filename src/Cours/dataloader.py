from pathlib import Path
import logging
import pickle
from datetime import datetime
from typing import Sequence
import requests
import io

import pandas as pd
import yfinance as yf

# CORRECTION 1: Import de PriceSeries (à adapter selon votre structure)
from priceseries import PriceSeries

class DataLoader:
    """
    Charge des données de marché depuis Yahoo Finance avec un système de cache.
    
    Le système de cache gère cinq scénarios de correspondance temporelle :
    1. EXACT : La requête correspond exactement aux données en cache
    2. CONTAINS : La requête est un sous-ensemble du cache
    3. OVERLAP_AFTER : Intersection partielle, fetch complémentaire à droite
    4. OVERLAP_BEFORE : Intersection partielle, fetch complémentaire à gauche
    5. MISS : Aucune donnée en cache, fetch complet nécessaire
    
    Attributes:
        cache_dir: Répertoire de stockage du cache
        logger: Logger pour le suivi des opérations
    
    """
    
    def __init__(self, cache_dir: str = ".cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_cache_path(
        self, 
        ticker: str, 
        price_col: str, 
        dates: tuple[str, str]
    ) -> Path:
        """
        Génère le chemin du fichier cache pour une requête donnée.
        
        Format: {ticker}_{price_col}_{start}_{end}.pkl
        """
        return self.cache_dir / f"{ticker}_{price_col}_{dates[0]}_{dates[1]}.pkl"

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

    def _load_from_cache(
        self,
        ticker: str,
        price_col: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> tuple[pd.DataFrame | None, str, tuple | None]:
        """
        Recherche et charge les données disponibles en cache.
        
        Parcourt les fichiers du répertoire cache pour trouver une correspondance
        avec le couple (ticker, price_col) et détermine le type de chevauchement.
        
        Args:
            ticker: 
            price_col: Nom de la colonne prix ('Close', 'Open', etc.)
            start_date: Date de début de la requête
            end_date: Date de fin de la requête
        
        Returns:
            tuple: (dataframe, status, gap_range)
            - dataframe: Données en cache ou None
            - status: Type de correspondance
            - gap_range: (gap_start, gap_end) si overlap, sinon None
        """
        if not self.cache_dir.exists():
            return (None, "miss", None)

        # Itération sur les fichiers du cache pour match (ticker, price_col)
        for file_path in self.cache_dir.iterdir():
            if not file_path.is_file() or file_path.suffix != '.pkl':
                continue

            try:
                # Parse le nom du fichier
                name_parts = file_path.stem.split('_')
                
                # Vérification du format attendu
                if len(name_parts) < 4:
                    continue
                
                cached_ticker = name_parts[0]
                cached_col = name_parts[1]
                cached_start_str = name_parts[2]
                cached_end_str = name_parts[3]

                # Vérifier la correspondance ticker + price_col
                if cached_ticker != ticker or cached_col != price_col:
                    continue

                # Parser les dates
                cached_start = pd.to_datetime(cached_start_str)
                cached_end = pd.to_datetime(cached_end_str)

                # Déterminer le type d'overlap
                status, gap_start, gap_end = self._check_date_overlap(
                    cached_start, cached_end, start_date, end_date
                )

                if status != "miss":
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)

                    # Reconstruire le DataFrame avec les dates
                    prices_list = data['prices']
                    dates_list = data.get('dates')
                    
                    df = pd.DataFrame({price_col: prices_list})
                    
                    if dates_list is not None:
                        # Utiliser les dates réelles stockées
                        df.index = pd.to_datetime(dates_list)
                    else:
                        # Fallback: utiliser les jours ouvrés
                        date_range = pd.bdate_range(
                            start=cached_start, 
                            periods=len(df)
                        )
                        df.index = date_range

                    if status == "exact":
                        return (df, "exact", None)
                    elif status == "contains":
                        return (df, "contains", None)
                    elif status.startswith("overlap"):
                        return (df, status, (gap_start, gap_end))

            except (ValueError, KeyError, pickle.UnpicklingError) as e:
                # Ignorer les fichiers cache corrompus
                self.logger.warning(f"Fichier cache corrompu {file_path}: {e}")
                continue

        return (None, "miss", None)
    
    def _save_to_cache(
        self, 
        cache_path: Path, 
        prices: list[float],
        dates: list,
        ticker: str, 
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
            "fetched_at": datetime.now().isoformat(),
            "n_prices": len(prices),
            "prices": prices,
            "dates": dates
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        self.logger.debug(f"Cache sauvegardé: {cache_path}")
    
    def _fetch_from_yfinance(
        self,
        ticker: str,
        price_col: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> pd.DataFrame | None:
        """
        Fetch les données depuis Yahoo Finance.
        
        Returns:
            DataFrame avec les prix ou None si échec
        """
        try:
            ticker_instance = yf.Ticker(ticker)
            df = ticker_instance.history(start=start_date, end=end_date)
            
            if df.empty:
                self.logger.warning(f"Dataframe vide pour {ticker}")
                return None
            
            if price_col not in df.columns:
                self.logger.error(f"{price_col} n'est pas dans le Dataframe de {ticker}")
                raise KeyError(f"{price_col} n'est pas dans le Dataframe de {ticker}")
            
            return df[[price_col]]  # Retourner DataFrame avec une seule colonne
            
        except Exception as e:
            self.logger.error(f"Erreur lors du fetch de {ticker}: {e}")
            return None
    
    # CORRECTION 2: Réécriture complète avec utilisation du cache
    def fetch_single_ticker(
        self, 
        ticker: str, 
        price_col: str, 
        dates: tuple[str, str]
    ) -> PriceSeries | None:
        """
        Récupère les données de prix d'un ticker unique avec système de cache.
        
        Args:
            ticker: Symbole (ex: 'AAPL')
            price_col: Nom de la colonne prix (ex: 'Close', 'Open')
            dates: (start_date, end_date) au format 'YYYY-MM-DD'
        
        Returns:
            Instance de PriceSeries ou None si échec
        """
        start_date = pd.Timestamp(dates[0])
        end_date = pd.Timestamp(dates[1])
        
        # Étape 1: Vérifier le cache
        cached_df, status, gap_range = self._load_from_cache(
            ticker, price_col, start_date, end_date
        )
        
        final_df = None
        
        # Étape 2: Traiter selon le statut du cache
        if status == "exact":
            self.logger.info(f"Cache HIT (exact) pour {ticker}")
            final_df = cached_df
            
        elif status == "contains":
            self.logger.info(f"Cache HIT (contains) pour {ticker}")
            # Filtrer pour la période demandée
            final_df = cached_df.loc[start_date:end_date]
            
        elif status.startswith("overlap"):
            self.logger.info(f"Cache HIT (partial: {status}) pour {ticker}")
            gap_start, gap_end = gap_range
            
            # Fetch les données manquantes
            gap_df = self._fetch_from_yfinance(ticker, price_col, gap_start, gap_end)
            
            if gap_df is not None:
                # Fusionner cache + nouvelles données
                if status == "overlap_after":
                    final_df = pd.concat([cached_df, gap_df]).sort_index()
                else:  # overlap_before
                    final_df = pd.concat([gap_df, cached_df]).sort_index()
                
                # Sauvegarder la période complète
                self._save_complete_range(ticker, price_col, final_df, dates)
            else:
                # Si fetch échoue, utiliser ce qu'on a en cache
                final_df = cached_df
                
        else:  # status == "miss"
            self.logger.info(f"Cache MISS pour {ticker}, fetch complet")
            final_df = self._fetch_from_yfinance(ticker, price_col, start_date, end_date)
            
            if final_df is not None:
                # Sauvegarder en cache
                cache_path = self._get_cache_path(ticker, price_col, dates)
                prices_list = final_df[price_col].tolist()
                dates_list = final_df.index.tolist()
                
                self._save_to_cache(
                    cache_path, prices_list, dates_list, 
                    ticker, dates[0], dates[1]
                )
        
        # Étape 3: Convertir en PriceSeries
        if final_df is None or final_df.empty:
            self.logger.warning(f"Aucune donnée disponible pour {ticker}")
            return None
        
        # CORRECTION 3: Convertir Series pandas en list et utiliser ticker comme nom
        prices_list = final_df[price_col].tolist()
        
        return PriceSeries(values=prices_list, name=ticker)
    
    def _save_complete_range(
        self,
        ticker: str,
        price_col: str,
        df: pd.DataFrame,
        dates: tuple[str, str]
    ) -> None:
        """
        Sauvegarde une plage de données complète après fusion cache + fetch.
        """
        cache_path = self._get_cache_path(ticker, price_col, dates)
        prices_list = df[price_col].tolist()
        dates_list = df.index.tolist()
        
        self._save_to_cache(
            cache_path, prices_list, dates_list,
            ticker, dates[0], dates[1]
        )
    
    def fetch_multiple_tickers(
        self,
        tickers: Sequence[str],
        price_col: str,
        dates: tuple[str, str]
    ) -> dict[str, PriceSeries]:
        """
        Récupère les données de prix pour plusieurs tickers.
        
        Returns:
            Dictionnaire {ticker: PriceSeries}
        """
        results = {}
        for ticker in tickers:
            ps = self.fetch_single_ticker(ticker, price_col, dates)
            if ps is not None:
                results[ticker] = ps
        return results
    
    # CORRECTION 4: Implémentation de clear_cache
    def clear_cache(self) -> int:
        """
        Supprime tous les fichiers du cache.
        
        Returns:
            Nombre de fichiers supprimés
        """
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file() and file_path.suffix == '.pkl':
                try:
                    file_path.unlink()
                    count += 1
                    self.logger.debug(f"Supprimé: {file_path}")
                except Exception as e:
                    self.logger.error(f"Erreur lors de la suppression de {file_path}: {e}")
        
        self.logger.info(f"Cache nettoyé: {count} fichier(s) supprimé(s)")
        return count

if __name__ == "__main__":
    Dataloader = DataLoader(cache_dir=".cache")
    result=Dataloader.fetch_single_ticker("AAPL", "Close", ("2022-01-01", "2024-06-01"))
    print(result.values)
    pass