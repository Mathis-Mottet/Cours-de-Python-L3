import math
 
class PriceSeries:
    """
    Représentation d'une série temporelle de prix financiers.
   
    Attributes:
        values: Liste de prix indexés par le temps
        name: Identifiant de la série
   
    Class Attributes:
        TRADING_DAYS_PER_YEAR: Constante d'annualisation
        (convention US equities, peut varier selon l'actif)
    """
    TRADING_DAYS_PER_YEAR = 252
    def __init__(self, values: list[float], name: str | None) -> None:
        self.name = name 
        self.values = list(values) # Copie défensive
 
    def __repr__(self):
        """Représentation pour les développeurs (debugging)."""
        return f"PriceSeries({self.name!r}, {self.values!r})" 
   
    def __str__(self): 
        """Représentation pour les utilisateurs."""
        if self.values: 
            return f"{self.name}: {self.values[-1]: .2f} (latest)" 
        return f"{self.name}: empty"
   
    def __len__(self):
        return len(self.values) 
   
    def get_linear_return(self, t):
        """
        Calcul le rendement linéaire (non-ajusté des dividendes) entre l'index t et t-1.
        Addifitif (avec un pondération) entre actifs pour une cross-section des données d'un portefeuille.
       
        Args:
            t (int): position temporelle de la valeur
 
        Returns:
            float:
        """
        if t < 1:
            raise ValueError(f"t doit être >= 1, reçu: {t}")
        if t >= len(self.values):
            raise IndexError(f"t={t} hors limites (max:{len(self.values)-1})")
        return (self.values[t] - self.values[t-1]) / self.values[t-1]
   
    def get_log_return(self, t):
        """
        Calcul le log-rendement entre l'index t et t-1
        Addifitif dans le temps pour un actif.
        Args:
            t (int): position temporelle de la valeur
 
        Returns:
            float:
        """
        return math.log(self.values[t]/self.values[t-1])
   
    @property
    def total_return(self) -> float:
        """
        Calcule le rendement linéaire total de
        l'actif sans prise en compte de réinvestissement (buy-and-hold)
       
        Returns:
            float:
        """
        return (self.values[-1] - self.values[0]) / self.values[0]
 
    def get_all_linear_returns(self) -> list[float]:
        """Retourne la liste de tous les rendements linéaires de la série de prix.
 
        Returns:
            list[float]:
        """
        return [self.get_linear_return(t) for t in range(1, len(self.values))]
   
    def get_all_log_returns(self) -> list[float]:
        """Retourne la liste de tous les rendements
 
        Returns:
            list[float]:
        """
        return [self.get_log_return(t) for t in range(1, len(self.values))]
   
    def get_annualized_volatility(self) -> float:
        """
        Volatilité annualisée à partir des log-rendements.
        Formule: σ_annual = σ_daily × √252
        Le scaling fonctionne sous l'hypothèse rendements IID.
        Cette hypothèse n'est pas respectée pour les rendements
        (voir faits stylisés classiques de la vol comme le clustering)
        Pour une meilleure estimation, il faudrait considérer par exemple:
        - les modèles GARCH
        - une moyenne mobile exponentielle
        """
        if len(self.values) < 3: 
            raise ValueError("Not enough data points") 
 
        log_returns = self.get_all_log_returns()
        n = len(log_returns)
        mean = sum(log_returns) / n
        var = sum((l_r - mean)**2 for l_r in log_returns) / (n - 1) 
        daily_vol = math.sqrt(var) 
 
        return daily_vol * math.sqrt(self.TRADING_DAYS_PER_YEAR)
   
    def get_annualized_return(self) -> float:
        """
        Retourne le rendement annualisé sur toute la période.
        """
        if len(self) < 2:
            raise ValueError("Not enough data points")
        r = self.get_all_log_returns()
        return (sum(r) / len(r)) * self.TRADING_DAYS_PER_YEAR
   
    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Ratio de sharpe annualisé :
            - ratio entre les rendements esperés d'une stratégie et sa vol
            - rendement par unité de risque
       
        Formule: SR = (μ - r_f) / σ
       
        Args:
            risk_free_rate: taux sans risque annuel
       
        """
        vol =  self.get_annualized_volatility()
        if vol == 0:
            raise ValueError("Vol is equal to zero")
        excess_return = self.get_annualized_return() - risk_free_rate
        return excess_return / vol
   
    def drawdown_at(self, t: int) -> float:
        """
        Retourne le drawdown à l'instant t depuis le début de la série.
        Mesure le déclin par rapport à un pique historique.
 
        Args:
            t (int): index de position de la valeur supérieure de l'intervalle considéré.
 
        Returns:
            float: drawdown (valeur négative ou nulle)
        """
        if t < 0 or t >= len(self.values):
            raise IndexError(f"index {t} is out of range for series of length {len(self.values)}")
 
        peak = max(self.values[:t+1])
        if peak == 0:
            return 0.0
        return (self.values[t] - peak) / peak
 
    def max_drawdown(self) -> float:
        """
        Retourne le drawdown maximum sur toute la série.
 
        Returns:
            float: drawdown maximum (valeur négative ou nulle)
        """
        if len(self.values) < 2:
            raise ValueError("Not enough values to calculate drawdown")
 
        max_dd = 0.0
        peak = self.values[0]
 
        for value in self.values[1:]:
            peak = max(peak, value)
            if peak > 0:
                dd = (value - peak) / peak
                max_dd = min(max_dd, dd)
 
        return max_dd