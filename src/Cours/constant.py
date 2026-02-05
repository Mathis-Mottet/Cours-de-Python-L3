from enum import StrEnum

class CurrencyEnum(StrEnum):
    """Enumeration des devis supportées"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"