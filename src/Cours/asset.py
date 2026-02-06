import numpy as np
from Cours.priceseries import PriceSeries
from Cours.constant import CurrencyEnum

class Asset:
    """
    Représente un actif financier avec son historique de prix.
   
    Pattern de conception : COMPOSITION
    ───────────────────────────────────
    Asset POSSÈDE une PriceSeries (relation HAS-A, pas IS-A).
   
    Le cycle de vie de PriceSeries est lié à celui de Asset :
    - Créé quand Asset est créé
    - Détruit quand Asset est détruit
   
    Attributes:
        ticker: Ticker (ex: 'AAPL')
        prices: Instance PriceSeries contenant l'historique (COMPOSÉE)
        sector: Classification sectorielle
        currency: Devise des prix (défaut: USD)
    """
   
    def __init__(
        self,
        ticker: str,
        prices: PriceSeries,
        sector: str | None = None,
        currency: CurrencyEnum = CurrencyEnum.USD
    ) -> None:
        """
        Initialise un Asset.
       
        Args:
            ticker: Symbole boursier (ne peut pas être vide)
            prices: Série de prix (ne peut pas être vide)
            sector: Secteur d'activité (optionnel)
            currency: Devise (défaut: USD)
       
        Raises:
            ValueError: Si ticker est vide ou prices est vide
        """
        # Validation des entrées dans le constructeur
        if not ticker or not ticker.strip():
            raise ValueError("Le ticker ne peut pas être vide")
        if len(prices) == 0:
            raise ValueError("La série de prix ne peut pas être vide")
        if any(p < 0 for p in prices.values):
            raise ValueError("La série de prix ne peut pas contenir de valeurs négatives")
        
        self.ticker = ticker.upper()  # Normalisation en majuscules
        self.prices = prices  # Composition : Asset POSSÈDE une PriceSeries
        self.sector = sector
        self.currency = currency
   
    def __repr__(self) -> str:
        """Représentation pour le développement."""
        return f"Asset({self.ticker!r}, {len(self.prices)} prices)"
   
    def __str__(self) -> str:
        """Représentation pour l'utilisateur."""
        return f"{self.ticker}: ${self.current_price:.2f}"
   
    @property
    def current_price(self) -> float:
        """Dernier prix connu."""
        return self.prices.values[-1]
   
    @property
    def volatility(self) -> float:
        """Volatilité annualisée (délègue à PriceSeries)."""
        return self.prices.get_annualized_volatility()
   
    @property
    def total_return(self) -> float:
        """Rendement total (délègue à PriceSeries)."""
        return self.prices.total_return
   
    @property
    def sharpe_ratio(self) -> float:
        """Ratio de Sharpe (délègue à PriceSeries)."""
        return self.prices.sharpe_ratio()
   
    @property
    def max_drawdown(self) -> float:
        """Drawdown maximum (délègue à PriceSeries)."""
        return self.prices.max_drawdown()
    
    def correlation_with(self, other: "Asset") -> float:
        """
        Calcule la corrélation de Pearson des log-rendements avec unautre actif
        
        Args:
            other: Un autre Asset

        Returns:
            Coefficient de corrélation entre -1 et 1
        """

        x = self.prices.get_all_log_returns()
        y = other.prices.get_all_log_returns()

        n = min(len(x), len(y)) # On prend le plus petit vecteur

        if n < 2:
            raise ValueError(
                f"Pas assez d'observations communes: {n}. "
                "Minimum requis: 2."
            )


        x, y = x[:n], y[:n] # On ajuste x et y pour qu'ils soient de même taille

        mean_x = sum(x)/n
        mean_y = sum(y)/n

        cx = [xi - mean_x for xi in x] # Crochet [ ] car list
        cy = [yi - mean_y for yi in y]

        covariance = sum(cx_i * cy_i for cx_i, cy_i in zip(cx, cy)) / (n-1)
        var_x = sum(cx_i * cx_i for cx_i in cx) / (n-1)
        var_y = sum(cy_i * cy_i for cy_i in cy) / (n-1)
        
        if var_x == 0 or var_y == 0:
            raise ValueError(
                "Variance nulle détectée. "
                "La corrélation n'est pas définie pour une série constante."
            )
        
        correlation = covariance / (var_x**(1/2) * var_y**(1/2))

        return correlation
    
    
    def correlation_with_numpy(self, other: "Asset") -> float:
        """
        Calcule la corrélation de Pearson des log-rendements avec un autre actif.
        
        Args:
            other: Un autre Asset
        
        Returns:
            Coefficient de corrélation entre -1 et 1
        """
        # Récupération des log-rendements
        x = np.array(self.prices.get_all_log_returns())
        y = np.array(other.prices.get_all_log_returns())
        
        # Alignement des longueurs (gestion des séries de tailles différentes)
        n = min(len(x), len(y))
        if n < 2:
            raise ValueError(
                f"Pas assez d'observations communes: {n}. "
                "Minimum requis: 2."
            )
        
        x = x[:n]
        y = y[:n]
        
        # Centrage des valeurs (soustraction de la moyenne)
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        
        # Covariance (numérateur)
        covariance = np.dot(x_centered, y_centered) / (n - 1)
        
        # Variances pour le dénominateur
        var_x = np.dot(x_centered, x_centered) / (n - 1)
        var_y = np.dot(y_centered, y_centered) / (n - 1)
        
        # Vérification de la variance nulle
        if var_x == 0 or var_y == 0:
            raise ValueError(
                "Variance nulle détectée. "
                "La corrélation n'est pas définie pour une série constante."
            )
        
        correlation = covariance / np.sqrt(var_x * var_y)

        return correlation